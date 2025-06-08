# CV_stable

Dreambooth on Stable Diffusion
This is an implementtaion of Google's Dreambooth with Stable Diffusion. The original Dreambooth is based on Imagen text-to-image model. However, neither the model nor the pre-trained weights of Imagen is available. To enable people to fine-tune a text-to-image model with a few examples, I implemented the idea of Dreambooth on Stable diffusion.

This code repository is based on that of Textual Inversion. Note that Textual Inversion only optimizes word ebedding, while dreambooth fine-tunes the whole diffusion model.

The implementation makes minimum changes over the official codebase of Textual Inversion. In fact, due to lazyness, some components in Textual Inversion, such as the embedding manager, are not deleted, although they will never be used here.

Update
9/20/2022: I just found a way to reduce the GPU memory a bit. Remember that this code is based on Textual Inversion, and TI's code base has this line, which disable gradient checkpointing in a hard-code way. This is because in TI, the Unet is not optimized. However, in Dreambooth we optimize the Unet, so we can turn on the gradient checkpoint pointing trick, as in the original SD repo here. The gradient checkpoint is default to be True in config. I have updated the codes.

Usage
Preparation
First set-up the ldm enviroment following the instruction from textual inversion repo, or the original Stable Diffusion repo.

To fine-tune a stable diffusion model, you need to obtain the pre-trained stable diffusion models following their instructions. Weights can be downloaded on HuggingFace. You can decide which version of checkpoint to use, but I use sd-v1-4-full-ema.ckpt.

We also need to create a set of images for regularization, as the fine-tuning algorithm of Dreambooth requires that. Details of the algorithm can be found in the paper. Note that in the original paper, the regularization images seem to be generated on-the-fly. However, here I generated a set of regularization images before the training. The text prompt for generating regularization images can be photo of a <class>, where <class> is a word that describes the class of your object, such as dog. The command is

python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt --prompt "a photo of a <class>" 
I generate 8 images for regularization, but more regularization images may lead to stronger regularization and better editability. After that, save the generated images (separately, one image per .png file) at /root/to/regularization/images.

Updates on 9/9 We should definitely use more images for regularization. Please try 100 or 200, to better align with the original paper. To acomodate this, I shorten the "repeat" of reg dataset in the config file.

For some cases, if the generated regularization images are highly unrealistic (happens when you want to generate "man" or "woman"), you can find a diverse set of images (of man/woman) online, and use them as regularization images.

Training
Training can be done by running the following command

python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word <xxx>
Detailed configuration can be found in configs/stable-diffusion/v1-finetune_unfrozen.yaml. In particular, the default learning rate is 1.0e-6 as I found the 1.0e-5 in the Dreambooth paper leads to poor editability. The parameter reg_weight corresponds to the weight of regularization in the Dreambooth paper, and the default is set to 1.0.

Dreambooth requires a placeholder word [V], called identifier, as in the paper. This identifier needs to be a relatively rare tokens in the vocabulary. The original paper approaches this by using a rare word in T5-XXL tokenizer. For simplicity, here I just use a random word sks and hard coded it.. If you want to change that, simply make a change in this file.

Training will be run for 800 steps, and two checkpoints will be saved at ./logs/<job_name>/checkpoints, one at 500 steps and one at final step. Typically the one at 500 steps works well enough. I train the model use two A6000 GPUs and it takes ~15 mins.

Generation
After training, personalized samples can be obtained by running the command

python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /path/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>" 
In particular, sks is the identifier, which should be replaced by your choice if you happen to change the identifier, and <class> is the class word --class_word for training.
