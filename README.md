# DCQA
Hi all, this is the official repository for ECAI 2024 paper: Differentiating Choices via Commonality for Multiple-Choice Question Answering. Our paper can be found at [arXiv link]. We sincerely apprecaite your interests in our projects!

## Architecture
Brifely, we propose a novel MCQA model by differentiating choices through identifying and eliminating their commonality, called DCQA. 
![image](architecture.png)

## Dependencies
The main libraries we use as follows.
* transformer==4.38.2
* tqdm==4.65.0
* torch==2.0.0
* rouge==1.0.1
* numpy==1.24.2

## Dataset
If the information in the dataset is incomplete, please download them in their official website.
* [CAQA](https://huggingface.co/datasets/tau/commonsense_qa)
* [OBQA](https://huggingface.co/datasets/allenai/openbookqa)
* [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
* [QASC](https://huggingface.co/datasets/allenai/qasc)
* [PIQA](https://huggingface.co/datasets/ybisk/piqa)
* [SocialIQA](https://huggingface.co/datasets/allenai/social_i_qa)
  
## Run
Before running the code, please download the language model (T5-Base, T5-Large, Unifiedqa-T5-Base and Unifiedqa-T5-Large).
* [T5-Base](https://huggingface.co/google-t5/t5-base/tree/main)
* [T5-Large](https://huggingface.co/google-t5/t5-large/tree/main)
* [Unifiedqa-T5-Base](https://huggingface.co/allenai/unifiedqa-t5-base/tree/main)
* [Unifiedqa-T5-Large](https://huggingface.co/allenai/unifiedqa-t5-large/tree/main)

Then run this commond to begin trainning.
* python run_genmc.py --model_path DATA_PATH --choice_num Choice_Num --data_path_train TRAIN_FILE --data_path_dev DEV_FILE --data_path_test TEST_FILE 

Taking the t5-base as language model, run the five datasets as follows.
* python run_genmc.py --model_path t5-base --choice_num 5 --data_path_train ./data/csqa/in_hourse/train.jsonl  --data_path_dev ./data/csqa/in_hourse/dev.jsonl  --data_path_test ./data/csqa/in_hourse/test.jsonl
* python run_genmc.py --model_path t5-base --choice_num 4 --data_path_train ./data/obqa/train.jsonl  --data_path_dev ./data/obqa/dev.jsonl  --data_path_test ./data/obqa/test.jsonl
* python run_genmc.py --model_path t5-base --choice_num 4 --data_path_train ./data/arc_easy/in_hourse/train.jsonl  --data_path_dev ./data/arc_easy/in_hourse/dev.jsonl  --data_path_test ./data/arc_easy/in_hourse/test.jsonl
* python run_genmc.py --model_path t5-base --choice_num 4 --data_path_train ./data/arc_challenge/in_hourse/train.jsonl  --data_path_dev ./data/arc_challenge/in_hourse/dev.jsonl  --data_path_test ./data/arc_challenge/in_hourse/test.jsonl
* python run_genmc.py --model_path t5-base --choice_num 8 --data_path_train ./data/qasc/in_hourse/train.jsonl  --data_path_dev ./data/qasc/in_hourse/dev.jsonl  --data_path_test ./data/qasc/in_hourse/test.jsonl



## Reference
