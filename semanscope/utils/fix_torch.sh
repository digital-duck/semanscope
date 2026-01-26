conda activate zinets
pip uninstall torchvision transformers sentence-transformers -y
pip install torch==2.9.0 torchvision==0.19.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.45.0 sentence-transformers==5.1.2
pip install transformers --upgrade