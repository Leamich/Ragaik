# Ragaik: A Math-Specific QA System Based on OCR

\[ [pdf](https://drive.google.com/file/d/1Yo-lEBIysRfTfn74YHYbhpOFCDvaOhnd/view?usp=drive_link) \] \[ [demostration](https://drive.google.com/file/d/1Iy2KbOZtyrLeM5f-nf-vt1imODo9WvN3/view?usp=drive_link) \]

_Currently in development_

**Abstract**. This project aims to create a QA system based on a database of handwritten mathematical documents. It will include features for building a vectorized text database using OCR, and for deploying an on-demand server with an interface to interact with the QA system.

**Authors**: Michael Leontjev ([@Leamich](https://github.com/Leamich)), Konstantin Anisimov ([@TheKostins](https://github.com/TheKostins)), Sergey Shorohov ([@le9endwp](https://github.com/le9endwp/)), Pavel Seroglazov ([@ubetu](https://github.com/ubetu)).

The project was created as a first-year bachelor’s project at HSE Saint Petersburg in the Applied Data Analysis and Artificial Intelligence program.

# Setting up environment and launching the project (obviously in UNIX environment)

## Tools needed 

1. uv [(you can see the installation instructions here)](https://docs.astral.sh/uv/getting-started/installation/)
2. yarn
2. docker (you know how to install this)
3. [(ollama)](https://ollama.com/)

## Setup project

1. Clone the repo `git clone https://github.com/Leamich/Ragaik`
2. Setup venv 
```bash 
uv venv
source .venv/bin/activate
uv sync
```
3. Create retriever cache RUN `python -m RAG.make_retriever_cache` from the root
4. Build the frontend 
```bash
cd web/frontend/ 
yarn install
```
5. Run `ollama pull phi4`

## Start the project
1. Start the redis `docker compose up`
2. Start the backend server `fastapi dev RAG/app.py` (you may also want to look at `RAG/config.py` for the environment variables to set)
3. Start the frontend `cd web/frontend/ && yarn dev `

