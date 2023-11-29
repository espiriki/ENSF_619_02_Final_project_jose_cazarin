# Offical Git Repo for the paper "Combining Natural Language and Images for Garbage Classification: A Public Benchmark"

## Instructions on how to repo the results in the paper
First, download the dataset set. Unzip the folders **CVPR_2024_dataset_Train**, **CVPR_2024_dataset_Val**, **CVPR_2024_dataset_Test** it into the root folder of this repo.

Then, to repro the results in the test set, download the saved model weights and extract the *.pth files into the folder **saved_model_weights** of this repo.

To repro the training process of the **EfficientNet B0**, use the following command:

    python main_image.py --image_model=b0 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=70 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet B4**, use the following command:

    python main_image.py--dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --image_model=b4 --epochs=30 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet B5**, use the following command:

    python main_image.py --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --image_model=b5 --epochs=40 --balance_weights --reg=0.1 

To repro the training process of the **EfficientNet V2 Small**, use the following command:

    python main_image.py --image_model=eff_v2_small --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet V2 Medium**, use the following command:

    python main_image.py --image_model=eff_v2_medium -dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=40 --balance_weights --reg=0.1 --prob_aug=0.9

To repro the training process of the **EfficientNet V2 Large**, use the following command:

    python main_image.py --image_model=eff_v2_large --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **ResNet 18**, use the following command:

    python main_image.py --image_model=res18 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=40 --opt=adamw --balance_weights --reg=0.1

To repro the training process of the **ResNet 50**, use the following command:

    python main_image.py --image_model=res50 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=40 --opt=adamw --balance_weights --reg=0.1

To repro the training process of the **ResNet 152**, use the following command:

    python main_image.py --image_model=res152 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=60 --opt=adamw --balance_weights --reg=0.1 --prob_aug=0.6

To repro the training process of the **MobileNet V3-large**, use the following command:

    python main_image.py --image_model=mb --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=50 --balance_weights --reg=0.1

To repro the training process of the **ConvNeXt Base**, use the following command:

    python main_image.py --image_model=convnext --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **ViT-B/16**, use the following command:

    python main_image.py --image_model=transformer_B16 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **ViT-L/16**, use the following command:

    python main_image.py --image_model=transformer_L16 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **Distilbert**, use the following command:

    python main_text.py --text_model=distilbert --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=45 --balance_weights --reg=0.1

To repro the training process of the **RoBERTa**, use the following command:

    python main_text.py --text_model=roberta --dataset_folder_name=CVPR_2024_dataset --ft_epochs=60 --opt=adamw --epochs=60 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the **BART**, use the following command:

    python main_text.py --text_model=bart --dataset_folder_name=CVPR_2024_dataset --ft_epochs=30 --opt=adamw --epochs=10 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the **GPT2**, use the following command:

    python main_text.py --text_model=gpt2 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the multi-modal **EfficientNetV2 + Distilbert**, use the following command:

    python main_both.py --dataset_folder_name=CVPR_2024_dataset --ft_epochs=25 --fraction_lr=3 --image_prob_dropout=0.2 --late_fusion=classic --lr=0.01 --image_text_dropout=0.0 --epochs=45 --opt=sgd --reg=0.1 --balance_weights --acc_steps=10 --acc_steps_FT=10 --prob_aug=0.95 --model_dropout=0.6 --num_neurons_FC=256


The following commands will create an image with the confusion matrix and a .csv file with the test report in the folder **test_set_reports**.

To repro the results in the test set of the **EfficientNet B0** model , use the following command:



Download the development set [here](https://uofc-my.sharepoint.com/:u:/g/personal/josecarlos_cazarinfi_ucalgary_ca/EdvomJjyrB5Ektukwi2XkWoBPxyS-9ljBXfMKC2VJkkoug?e=8wiVwn), extract it and put on the same folder as in the `main.py` script

Download the test set [here](https://uofc-my.sharepoint.com/:u:/g/personal/josecarlos_cazarinfi_ucalgary_ca/EW94Adt6pK5HlLf11l5qOcEB0iRCvdVI2STxuxqwdWK6-g?e=4wQpoJ), extract it and put on the same folder as in the `main.py` script

