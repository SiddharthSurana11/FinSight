import streamlit as st

def display_signature():
    """
    Render FinSight Version 2 professional footer.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888888; padding: 15px 0px 10px 0px; font-size: 0.9rem; line-height: 1.6;'>
            <strong style='color: #7C4DFF; font-size: 1.05rem;'>FinSight Analytics Platform</strong> • Version 2.0<br/>
            Built with Python • Streamlit • Plotly • Yahoo Finance Market Engine<br/>
            <span style='font-size: 0.85rem; color: #AAAAAA;'>© 2026 Siddharth Surana. All Rights Reserved.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
