"""
Streamlit page for analyzing conversation history JSON files
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.conversation_analyzer import ConversationAnalyzer
from src.app.pages.conversation_analyzer_simplified import render_simplified_conversation


def render_message_analysis(analysis):
    """Render a single message analysis"""
    # Message header with risk badge
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Message {analysis.message_id + 1}** ({analysis.role})")
    
    with col2:
        if analysis.risk_level == 'high':
            st.error(f"⚠️ {analysis.risk_level.upper()}")
        elif analysis.risk_level == 'medium':
            st.warning(f"⚡ {analysis.risk_level.upper()}")
        else:
            st.success(f"✅ {analysis.risk_level.upper()}")
    
    # Message content
    with st.expander("Content", expanded=True):
        st.text(analysis.content[:200] + "..." if len(analysis.content) > 200 else analysis.content)
    
    # Detection results
    harmful_detections = []
    for detector, result in analysis.detections.items():
        if 'error' not in result and result.get('prediction') == '1':
            harmful_detections.append(detector)
    
    if harmful_detections:
        st.error(f"Issues detected: {', '.join(harmful_detections)}")
    
    # Recommendations
    if analysis.recommendations:
        with st.expander("Recommendations"):
            for rec in analysis.recommendations:
                st.info(f"• {rec}")
    
    st.divider()


def render_summary_stats(result):
    """Render summary statistics"""
    st.subheader("📊 Summary Statistics")
    
    # Message counts
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", result.total_messages)
    
    with col2:
        st.metric("AI Messages", result.ai_messages_analyzed)
    
    with col3:
        st.metric("User Messages", result.user_messages)
    
    with col4:
        total_issues = result.high_risk_messages + result.medium_risk_messages
        st.metric("Messages with Issues", total_issues)
    
    # Risk distribution
    st.subheader("🎯 Risk Distribution")
    
    risk_data = {
        'Risk Level': ['High', 'Medium', 'Low'],
        'Count': [result.high_risk_messages, result.medium_risk_messages, result.low_risk_messages]
    }
    
    df = pd.DataFrame(risk_data)
    st.bar_chart(df.set_index('Risk Level'))
    
    # Most common issues
    if result.summary['most_common_issues']:
        st.subheader("🔍 Most Common Issues")
        for issue in result.summary['most_common_issues']:
            st.warning(f"• {issue.replace('_', ' ').title()}")


def conversation_analyzer_page():
    """Conversation analyzer page"""
    st.title("💬 Conversation History Analyzer")
    st.markdown(
        "Analyze AI-human conversation histories for eldercare communication patterns. "
        "Upload a JSON file containing conversation messages to get started."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a conversation JSON file",
        type=['json'],
        help="JSON should contain a list of messages with 'role' and 'content' fields"
    )
    
    # Example format
    with st.expander("📝 Expected JSON Format"):
        st.code("""
[
    {
        "role": "user",
        "content": "おばあちゃん、今日のお薬飲んだ？"
    },
    {
        "role": "ai",
        "content": "さあ、一緒にお薬を飲みましょうね。"
    },
    {
        "role": "user", 
        "content": "もう飲んだの？"
    },
    {
        "role": "ai",
        "content": "これを全部飲んでから、そこに座ってください。"
    }
]
        """, language="json")
    
    if uploaded_file is not None:
        try:
            # Load JSON
            conversation_data = json.load(uploaded_file)
            
            st.success(f"✅ Loaded {len(conversation_data)} messages")
            
            # Analyze button
            if st.button("🔍 Analyze Conversation", type="primary", use_container_width=True):
                with st.spinner("Analyzing conversation..."):
                    # Initialize analyzer
                    analyzer = ConversationAnalyzer()
                    
                    # Run analysis
                    result = analyzer.analyze_conversation(conversation_data)
                    
                    # Save result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"conversation_analysis_{timestamp}.json"
                    output_path = Path("results") / "conversation_analyses" / output_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    analyzer.save_analysis_result(result, str(output_path))
                    
                    st.success(f"✅ Analysis complete! Results saved to {output_filename}")
                    
                    # Also generate simplified version
                    simplified_result = analyzer.analyze_conversation_simplified(conversation_data)
                    simplified_filename = f"conversation_analysis_simplified_{timestamp}.json"
                    simplified_path = Path("results") / "conversation_analyses" / simplified_filename
                    analyzer.save_simplified_result(simplified_result, str(simplified_path))
                    
                    # Store in session state for view switching
                    st.session_state['simplified_result'] = simplified_result
                    st.session_state['result'] = result
                    st.session_state['output_path'] = str(output_path)
                    st.session_state['simplified_path'] = str(simplified_path)
                    st.session_state['output_filename'] = output_filename
                    st.session_state['simplified_filename'] = simplified_filename
                    
                # Display results outside of button context
                if 'result' in st.session_state and 'simplified_result' in st.session_state:
                    st.header("Analysis Results")
                    
                    # Add toggle for view mode
                    view_mode = st.radio(
                        "View mode:",
                        ["Detailed", "Simplified"],
                        horizontal=True
                    )
                    
                    result = st.session_state['result']
                    simplified_result = st.session_state['simplified_result']
                    output_path = st.session_state.get('output_path')
                    simplified_path = st.session_state.get('simplified_path')
                    
                    if view_mode == "Detailed":
                        # Summary statistics
                        render_summary_stats(result)
                        
                        # Individual message analyses
                        st.subheader("📋 Message-by-Message Analysis")
                        
                        # Filter options
                        filter_col1, filter_col2 = st.columns(2)
                        
                        with filter_col1:
                            show_risk = st.multiselect(
                                "Show messages with risk level:",
                                ['high', 'medium', 'low'],
                                default=['high', 'medium']
                            )
                        
                        with filter_col2:
                            max_messages = st.slider(
                                "Max messages to display:",
                                min_value=1,
                                max_value=min(50, len(result.message_analyses)),
                                value=min(10, len(result.message_analyses))
                            )
                        
                        # Display filtered messages
                        displayed = 0
                        for analysis in result.message_analyses:
                            if analysis.risk_level in show_risk and displayed < max_messages:
                                render_message_analysis(analysis)
                                displayed += 1
                    else:
                        # Simplified view
                        try:
                            render_simplified_conversation(simplified_result)
                        except Exception as e:
                            st.error(f"Error rendering simplified view: {str(e)}")
                            st.error("Showing raw data instead:")
                            st.json({
                                'conversation': simplified_result.conversation,
                                'summary': simplified_result.summary,
                                'generated_at': simplified_result.generated_at
                            })
                    
                    # Download results
                    st.subheader("📥 Download Results")
                    
                    if output_path and simplified_path:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if Path(output_path).exists():
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    result_json = f.read()
                                
                                st.download_button(
                                    label="Download Detailed Results (JSON)",
                                    data=result_json,
                                    file_name=st.session_state.get('output_filename', 'detailed_result.json'),
                                    mime="application/json"
                                )
                        
                        with col2:
                            if Path(simplified_path).exists():
                                with open(simplified_path, 'r', encoding='utf-8') as f:
                                    simplified_json = f.read()
                                
                                st.download_button(
                                    label="Download Simplified Results (JSON)",
                                    data=simplified_json,
                                    file_name=st.session_state.get('simplified_filename', 'simplified_result.json'),
                                    mime="application/json"
                                )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your JSON file follows the expected format.")


if __name__ == "__main__":
    conversation_analyzer_page()