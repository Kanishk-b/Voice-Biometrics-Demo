import streamlit as st
import torch
import torchaudio
import soundfile as sf
import io
import warnings
from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings("ignore")

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Voice Verification API", page_icon="🎙️", layout="centered")

# --- AI ENGINE CACHING (CRITICAL FOR CLOUD) ---
@st.cache_resource(show_spinner="Booting AI Engine & Downloading Weights...")
def load_model():
    # This automatically downloads the weights from HuggingFace to the cloud server
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model"
    )

classifier = load_model()

# --- BIOMETRIC LOGIC (MEMORY-ONLY, NO FILES SAVED) ---
def process_audio_bytes(audio_bytes, max_duration=15):
    try:
        # Read the raw bytes straight from Streamlit's memory!
        numpy_signal, sr = sf.read(io.BytesIO(audio_bytes))
        
        if len(numpy_signal.shape) == 1:
            signal = torch.tensor(numpy_signal).unsqueeze(0).float()
        else:
            signal = torch.tensor(numpy_signal).transpose(0, 1).float()
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal)
            
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        signal = signal.squeeze()
        
        max_frames = 16000 * max_duration
        if signal.shape[0] > max_frames:
            signal = signal[:max_frames]
            
        batch = signal.unsqueeze(0)
        
        with torch.no_grad():
            embeddings = classifier.encode_batch(batch)
            
        return embeddings.squeeze()
    
    except Exception as e:
        raise RuntimeError(f"Audio processing failed. Is this a true .wav file? Error: {str(e)}")

# --- UI FRONTEND ---
st.title("🎙️ Voice Biometric Authentication")
st.write("Upload two audio files to mathematically verify if they belong to the same person.")

st.sidebar.header("⚙️ Engine Parameters")
threshold = st.sidebar.slider("Security Threshold", min_value=0.50, max_value=0.99, value=0.70, step=0.01)

col1, col2 = st.columns(2)
with col1:
    st.info("File 1 (Baseline)")
    file1 = st.file_uploader("Upload File 1", type=["wav"], key="f1", label_visibility="collapsed")
with col2:
    st.info("File 2 (Suspect)")
    file2 = st.file_uploader("Upload File 2", type=["wav"], key="f2", label_visibility="collapsed")

st.markdown("---")
if st.button("🔍 Verify Voices", type="primary", use_container_width=True):
    if file1 and file2:
        with st.spinner("Analyzing Biometric Voice Prints..."):
            try:
                # Send the raw memory bytes to the AI Engine
                emb1 = process_audio_bytes(file1.getvalue())
                emb2 = process_audio_bytes(file2.getvalue())
                
                # The Math
                score = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
                is_match = score >= threshold
                
                if is_match:
                    st.success(f"✅ **MATCH CONFIRMED** | Score: {score:.4f} (Threshold: {threshold})")
                else:
                    st.error(f"❌ **ACCESS DENIED: IMPOSTER** | Score: {score:.4f} (Threshold: {threshold})")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please upload both audio files.")