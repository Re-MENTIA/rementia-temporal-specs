"""
Simplified view rendering for conversation analyzer
"""

import streamlit as st


def render_simplified_conversation(simplified_result):
    """Render simplified conversation view"""
    
    # Summary at top
    st.subheader("📊 Quick Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Issues", simplified_result.summary['total_issues'])
    
    with col2:
        st.metric("High Risk", simplified_result.summary['high_risk'])
    
    with col3:
        st.metric("Medium Risk", simplified_result.summary['medium_risk'])
    
    with col4:
        if simplified_result.summary['most_common_issue']:
            st.metric("Most Common", simplified_result.summary['most_common_issue'].replace('_', ' ').title())
    
    # Conversation thread
    st.subheader("💬 Conversation Thread")
    
    # Create a simple view
    for msg in simplified_result.conversation:
        if msg['role'] == 'user':
            # User message
            st.markdown(f"**👤 User:** {msg['content']}")
        else:
            # AI message with detections
            risk_emoji = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢',
                'unknown': '⚪'
            }
            
            risk = msg.get('risk', 'unknown')
            st.markdown(f"**🤖 AI {risk_emoji[risk]}:** {msg['content']}")
            
            # Show detections as tags
            if 'detections' in msg:
                tags = []
                for detector, value in msg['detections'].items():
                    if value == '1':
                        # Shorten detector names
                        short_name = {
                            'terms_of_endearment': 'endearment',
                            'collective_instruction': 'collective',
                            'episode_memory': 'episode',
                            'open_end_question': 'open_end',
                            'long_speech': 'long',
                            'use_pronoun': 'pronoun'
                        }.get(detector, detector)
                        tags.append(f"`{short_name}:1`")
                
                if tags:
                    st.markdown("  " + " ".join(tags))
        
        st.divider()
    
    # Legend
    with st.expander("📖 Legend"):
        st.markdown("""
        **Risk Levels:**
        - 🔴 High risk
        - 🟡 Medium risk
        - 🟢 Low risk
        
        **Detection Tags:**
        - `endearment:1` - Terms of endearment detected
        - `collective:1` - Collective instruction detected
        - `episode:1` - Episode memory required
        - `open_end:1` - Open-ended question
        - `long:1` - Long/complex speech
        - `pronoun:1` - Vague pronoun usage
        """)