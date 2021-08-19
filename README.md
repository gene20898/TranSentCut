# Introduction
We created a new Thai sentence segmentation model called *TranSentCut - Transformer Based Thai Sentence Segmentation* . Here you can find code for training/evaluating the model.

# Training the model

## Setup working directory
After cloning the repo, *cd* into it and create the following directories: 

* checkpoints
* data
* logs
* models
* tmp

## Setup the environment
For perfect repeatablity with our results, we recommend using Nvidia Pytorch docker image. We used version 20.12 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12).

Pull the image
```
docker pull nvcr.io/nvidia/pytorch:20.12-py3
```

Start the container
```
docker run --gpus all -it --ipc=host --name <name> -v <host_path>:/<container_path> -p 6006:6006 nvcr.io/nvidia/pytorch:20.12-py3
```
where `<host_path>` is the path where you cloned this repo and `<container_path>` is any path inside the container you choose.

Setup the environment inside the container. Get inside the container and, *cd* to `<container_path>` and run
```
./install_requirements.sh
```
if the file is not executable, run *chmod +x install_requirements.sh* first.

## Get the training data, pretrained model and tokenizer
Go to https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main. Download `config.json` and `pytorch_model.bin` and put then in `models/wangchanberta-base-att-spm-uncased`. Then download `tokenizer_config.json`, `sentencepiece.bpe.model` and `sentencepiece.bpe.vocab` and put them in `models/tokenizer`. Rename `tokenizer_config.json` to `config.json`. The `models` directory now should look like this

```
models/
    tokenizer/
        config.json <--- tokenizer_config.json renamed
        sentencepiece.bpe.model
        sentencepiece.bpe.vocab
    wangchanberta-base-att-spm-uncased/
        config.json
        pytorch_model.bin
```

Next, get the ORCHID data from this Google Drive link: https://drive.google.com/drive/folders/1dD_uUX6de7cMohBVr52GUWqQylBH_dZM?usp=sharing. Put `xmlchid.xml` into `data` directory.

## Train the model
Inside the container at `<container_path>` from earlier, run
```
python train.py --config_path=config/TranSentCut.yaml
```
The result will be written to `tmp/experiment_results.txt`. Model will be saved to `models/version1`. The training parameters in `config/TranSentCut.yaml` is the best configurations we found. It should give final f1-score (macro) of 0.9296. Space-correct should be 0.9643.

# Evaluate the model

Once the model finished training, it can be evaluated on new data using
```
python eval.py --model_path=models/version1 --tokenizer_path=models/tokenizer --eval_data_path=<eval_path> --context_length=256
```
where `<eval_path>` is the path to the evaluation data (.txt). On out-of-domain data that is not similar to ORCHID, we got f1-score of 0.6903. Unfortunately we cannot release the evaluation data due to legal concerns. However it is relatively easy to construct your own evaluation data. To do so, create a text file, with one sentence per line and a blank line between paragraphs. Be sure to have two blank lines at the end of the file so that the last paragraph in the file is included in the evaluation.

The trained model is also available at https://drive.google.com/drive/folders/1y6ZSnh7N-BjT2stit917RMYnjwuSiJv7?usp=sharing if you just want to evaluate it. Replace `models/version1` in the above command with the path that you saved the model. Please get the tokenizer from https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main. Then rename the tokenizer files and setup the working directory according to the instruction in the training section.

# Docker images
We provide docker image for inference, please see `sumethy/TranSentCut` on Docker Hub.

# Reference 
(abstract only) TranSentCut − Transformer Based Thai Sentence Segmentation https://www.researchgate.net/publication/353996818_TranSentCut_-_Transformer_Based_Thai_Sentence_Segmentation. The full paper is under review.