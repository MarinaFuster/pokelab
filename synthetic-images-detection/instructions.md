# Human Setup Instructions

These are the steps you need to complete **before** handing this to the agent.

## 1. Get a HuggingFace account and token

1. Go to [huggingface.co](https://huggingface.co) and create an account if you don't have one
2. Go to **Settings → Access Tokens → New token**
3. Create a token with at least **Read** permissions
4. Copy the token value

## 2. Accept the ImageNet license

The real-image dataset requires you to personally accept the terms on HuggingFace.

1. Go to [huggingface.co/datasets/ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
2. Click **"Access repository"** and accept the license terms
3. Your HF account must be the same one whose token you use in step 1

## 3. Create the `.env` file

In the `synthetic-images-detection/` folder, create a file named `.env` with the following content:

```
HF_TOKEN=hf_your_token_here
```

Replace `hf_your_token_here` with the token you copied in step 1.

> The `.env` file should **not** be committed to git. Check that `.gitignore` covers it.

## 4. Set up a Python environment

The agent will install dependencies, but you need Python 3.10+ available. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

## 5. (Optional) GPU

The CLIP embedding step is significantly faster with a GPU. The experiment will run on CPU too, but expect ~30–60 minutes for embedding extraction on CPU vs. a few minutes on GPU. No code changes needed — PyTorch will auto-detect.

---

Once you've completed steps 1–4, hand control back to the agent and it will handle everything else.
