import os

import pandas as pd
import pydub
import streamlit as st
from joblib import load
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

import src.Classifiers as Classifiers
from src.FeatureExtractor import FeatureExtractor
from src.Recorder import Recorder
from src.ResultsManager import ResultsManager

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def patient_page(results_manager):
    recorder = Recorder()

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(204, 49, 49);
        width: 200px;
        height: 200px;
        border-radius: 50%;
    }
    </style>""", unsafe_allow_html=True)

    if st.button("Record"):
        with st.spinner(f"Say 'aaaaa' for 5 seconds..."):
            audio = recorder.record(5)
        
        try:
            feature_extractor = FeatureExtractor(audio.T)
            uid = results_manager.add_results(feature_extractor.get_features())
            results_manager.save()
            st.success(f"Thank you for your recording. Your UID is: {uid}. Please report to your doctor for further information.")
        except Exception as e:
            "Could not extract vocal features. Please try again."
            print(e)

def doctor_page(results_manager: ResultsManager):
    search = st.text_input("Search UID")
    dataframe_holder = st.empty()

    if search:
        status = results_manager.get_status(search)

        if status == 0:
            st.success(f"Patient {search} has not displayed vocal traits associated with Parkinson's disease.")
        elif status == 1:
            st.warning(f"Patient {search} has displayed vocal traits associated with Parkinsons's disease. Further analysis required.")
        else:
            st.error(f"Patient {search} does not yet have a prediction.")
        
        f"Features for patient {search}:"
        st.write(results_manager.to_dataframe(results_manager.get_features(search)))

        if st.button(f"Remove results for patient {search}"):
            results_manager.remove_results(search)
            results_manager.save()
            st.success(f"Results for patient {search} have been removed.")
    else:
        # st.write(results_manager.to_dataframe())
        dataframe_holder.dataframe(results_manager.to_dataframe())

        if st.button("Generate predictions"):
            # model = load('voting_classifier')
            model = Classifiers.SVM()
            unpredicted = results_manager.get_unpredicted()
            for uid in unpredicted:
                features = results_manager.get_features(uid)
                features = {k: v for k, v in features.items() if not pd.isnull(v)}
                features_dataframe = results_manager.to_dataframe(features)
                prediction = model.predict(features_dataframe)
                results_manager.set_status(uid, prediction[0])
            results_manager.save()
            dataframe_holder.dataframe(results_manager.to_dataframe())
            st.write("Predictions have been updated.")
            st.write(model.confusion_matrix)

def rtc_poc(results_manager):
    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        # desired_playing_state=st.session_state["recording"],
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True},
    )

    status_indicator = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)

            status_indicator.write("Please make a sustained 'aaaaaa' sound for around 5 seconds.")

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk

            # if len(st.session_state["audio_buffer"]) == 5000:
            #     st.session_state["recording"] = False
            #     break
        else:
            break

    audio_buffer = st.session_state["audio_buffer"]
    st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        audio_buffer.export("temp.wav", format="wav")
        feature_extractor = FeatureExtractor('temp.wav')
        uid = results_manager.add_results(feature_extractor.get_features())
        results_manager.save()
        os.remove('temp.wav')
    
        st.success(f"Thank you for your recording. Your UID is: {uid}. Please report to your doctor for further information.")


def main():
    st.title("Parkinson's Diagnostic Tool")

    pages = {
        "Patient view" : patient_page,
        "Patient view (WebRTC)" : rtc_poc,
        "Doctor view" : doctor_page,
    }

    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose view mode",
        page_titles
    )

    st.subheader(page_title)

    results_manager = ResultsManager('results/results.csv')
    page_func = pages[page_title]
    page_func(results_manager)

if __name__ == '__main__':
    main()
