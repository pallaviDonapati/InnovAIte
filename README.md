# 🚀 InnovAIte – AI-Powered Content Creation Platform

**InnovAIte** is an AI-powered platform designed to help users generate high-quality blog content and AI-generated images based on their input. It uses **LLaMA 2** for text generation and **Stable Diffusion** for image creation. Built with **Streamlit** and available on **Google Colab**, it simplifies and automates content creation for writers, marketers, and educators.

---

## 🏠 Homepage UI

Intuitive, minimal interface built using Streamlit for seamless interaction.

<div align="center">

<img src="assets/homepage%20UI.jpg" width="600"/>

</div>

---

## 📝 Text Generation

Generate personalized and high-quality articles with context-aware inputs.

<div align="center">

<img src="assets/Text%20Generation.jpg" width="400"/>
<img src="assets/Text%20Generation%20Prompt.jpg" width="400"/>
<br/>
<img src="assets/Text%20Generation%20Prompt%20Example.jpg" width="400"/>
<img src="assets/Text%20Generation%20Example.jpg" width="400"/>

</div>

---

## 🖼️ Image Generation

Create unique images from natural language prompts using Stable Diffusion.

<div align="center">

<img src="assets/Image%20Generation.jpg" width="400"/>
<img src="assets/Image%20Generation%20Prompt.jpg" width="400"/>
<br/>
<img src="assets/Stable%20Diffusion%20image.jpg" width="400"/>
</div>

---


## ✨ Features

- 📝 Blog generation using LLaMA 2 (via Hugging Face)
- 🎨 Image generation using Stable Diffusion (Diffusers)
- ⚙️ Prompt customization: tone, audience, style, and mood
- 📄 Export as Word Document (`.docx`)
- ☁️ Deployable on Streamlit and Google Colab

---

## 🧠 Tech Stack

| Component         | Technology                      |
|------------------|----------------------------------|
| Text Generation  | LLaMA 2 (13B Chat)               |
| Image Generation | Stable Diffusion v2.1            |
| UI               | Streamlit                        |
| Language         | Python                           |
| Platform         | Google Colab / Streamlit Cloud   |
| Libraries        | `transformers`, `diffusers`, `torch`, `streamlit`, `docx`, `bitsandbytes` |

---

## 🖥️ How to Run

### ✅ Run Locally

```bash
git clone https://github.com/pallaviDonapati/InnovAIte.git
cd InnovAIte
pip install -r requirements.txt
streamlit run entirescript.py
