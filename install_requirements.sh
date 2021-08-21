pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchtext==0.9.1
pip install bios==0.1.2
pip install fastapi==0.67.0
pip install gensim==4.0.1
pip install transformers==4.9.1
pip install ipywidgets==7.6.3
pip install pythainlp==2.3.1
pip install python-crfsuite==0.9.7
pip install pytorch-lightning==1.4.0
pip uninstall -y jupyter-tensorboard nvidia-tensorboard nvidia-tensorboard-plugin-dlprof tensorboard # comment this line if you setup new conda environment, leave it if you use nvidia-pytorch docker image
pip install tensorboard==2.4.0
pip install thai-segmenter==0.4.1