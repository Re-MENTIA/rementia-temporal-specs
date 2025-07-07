"""
Eldercare Communication Analyzer - Streamlit App
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.workflow import DetectionPipeline


# Page configuration
st.set_page_config(
    page_title="Eldercare Communication Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .detector-result {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load detection pipeline with optimized prompts if available"""
    # Check for optimized prompts
    prompts_path = "src/prompts/optimized_prompts.json"
    if Path(prompts_path).exists():
        st.sidebar.success("✓ Using optimized prompts")
        return DetectionPipeline(prompts_path)
    else:
        st.sidebar.info("ℹ Using default prompts")
        return DetectionPipeline()


def display_risk_badge(risk_level: str):
    """Display risk level badge with appropriate styling"""
    colors = {
        'high': '#f44336',
        'medium': '#ff9800', 
        'low': '#4caf50'
    }
    
    icons = {
        'high': '⚠️',
        'medium': '⚡',
        'low': '✅'
    }
    
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {colors[risk_level]};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-weight: bold;
            margin: 1rem 0;
        ">
            {icons[risk_level]} {risk_level.upper()} RISK
        </div>
        """,
        unsafe_allow_html=True
    )


def display_detector_result(name: str, detection: dict):
    """Display individual detector result"""
    # Define friendly names and descriptions
    detector_info = {
        'accommodation': {
            'name': 'Accommodation Speech',
            'icon': '💬',
            'description': 'Detects inappropriate terms of endearment'
        },
        'episode_memory': {
            'name': 'Episode Memory',
            'icon': '🧠',
            'description': 'Identifies questions requiring personal memory recall'
        },
        'open_end_question': {
            'name': 'Open-End Question',
            'icon': '❓',
            'description': 'Detects open-ended questions'
        },
        'long_speech': {
            'name': 'Long Speech',
            'icon': '📏',
            'description': 'Detects overly long or complex sentences'
        }
    }
    
    info = detector_info.get(name, {'name': name, 'icon': '📊', 'description': ''})
    
    # Determine if this is a positive detection
    is_positive = detection['prediction'] == '1'
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"### {info['icon']} {info['name']}")
        
    with col2:
        if is_positive:
            st.error(f"**Detected**: {detection['label']}")
        else:
            st.success(f"**Safe**: {detection['label']}")
            
        if info['description']:
            st.caption(info['description'])


def main():
    """Main application"""
    # Header
    st.title("🏥 Eldercare Communication Analyzer")
    st.markdown(
        "Analyze text for cognitive accessibility in eldercare settings. "
        "This tool detects communication patterns that may be challenging "
        "for individuals with cognitive impairments."
    )
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This system analyzes text across four dimensions:
            
            1. **Accommodation Speech**: Inappropriate language patterns
            2. **Episode Memory**: Questions requiring personal memory
            3. **Open-End Questions**: Questions needing elaborate responses
            4. **Long Speech**: Overly long or complex sentences
            
            **Risk Levels:**
            - 🟢 **Low**: Safe communication
            - 🟡 **Medium**: Could be improved
            - 🔴 **High**: May cause confusion or distress
            """
        )
        
        st.header("Examples")
        example_texts = {
            "Safe greeting": "こんにちは。今日はいい天気ですね。",
            "Inappropriate endearment": "おばあちゃん、今日も可愛いね！",
            "Episode memory question": "昨日の夕食で何を食べましたか？",
            "Open-ended + Episode": "先週の旅行はどうでしたか？詳しく聞かせてください。",
            "Closed preference": "コーヒーと紅茶、どちらがお好きですか？",
            "Long complex sentence": "今日は晴れて気温も高いので帽子をかぶって水分をしっかり取りませんか？",
            "Short simple": "水飲む？"
        }
        
        selected_example = st.selectbox(
            "Try an example:",
            options=list(example_texts.keys()),
            index=None,
            placeholder="Select an example..."
        )
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Main content area
    st.header("Input Text")
    
    # Text input
    if selected_example:
        input_text = st.text_area(
            "Enter text to analyze:",
            value=example_texts[selected_example],
            height=100,
            help="Enter any text you want to analyze for cognitive accessibility"
        )
    else:
        input_text = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="例: 昨日どこに行きましたか？",
            help="Enter any text you want to analyze for cognitive accessibility"
        )
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                # Run analysis
                result = pipeline.analyze(input_text)
            
            # Display results
            st.header("Analysis Results")
            
            # Risk level
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Risk Assessment")
                display_risk_badge(result.risk_level)
            
            with col2:
                st.subheader("Summary")
                if result.risk_level == 'high':
                    st.error(
                        "This communication pattern may cause confusion or distress "
                        "for individuals with cognitive impairments."
                    )
                elif result.risk_level == 'medium':
                    st.warning(
                        "This communication could be improved for better cognitive accessibility."
                    )
                else:
                    st.success(
                        "This communication follows good practices for cognitive accessibility."
                    )
            
            # Detailed detection results
            st.subheader("Detection Details")
            
            for name, detection in result.detections.items():
                if 'error' not in detection:
                    display_detector_result(name, detection)
                else:
                    st.error(f"{name}: {detection['error']}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if result.recommendations:
                for i, rec in enumerate(result.recommendations, 1):
                    if i == 1 and result.risk_level in ['high', 'medium']:
                        st.warning(rec)
                    else:
                        st.info(f"• {rec}")
            
            # Raw results (expandable)
            with st.expander("View Raw Results"):
                st.json({
                    'text': result.text,
                    'risk_level': result.risk_level,
                    'detections': result.detections,
                    'recommendations': result.recommendations
                })
                
        else:
            st.error("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.caption(
        "Built with ❤️ for improving eldercare communication. "
        "Based on research in cognitive accessibility."
    )


if __name__ == "__main__":
    main()