RESOLUTION=8
T=0.5

source /home/../../anaconda3/bin/activate  # Adjust the path to your conda installation
conda activate mv3dcd


for DATASET_NAME in Cantina Lounge Printing_area Lunch_room Meeting_room Garden Pots Zen Playground Porch; do
    for CLASSNAME in  Instance_1 Instance_2 ; do

        echo "Building reference scene: $DATASET_NAME, $CLASSNAME"
        python train.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME  --iterations 7000  --change $CLASSNAME  --resolution $RESOLUTION --checkpoint_iterations 7000 --save_iterations 7000

        echo "Rendering viewpoints for masks generation"
        python render_viewpoints.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME  --iterations 7000  --change $CLASSNAME  --resolution $RESOLUTION 

        echo "Creating noisy masks"
        python create_mask.py --t $T --input_folder output/PASLCD/$DATASET_NAME/$CLASSNAME

        echo "Learning the change channels and scene update"
        python train_masks.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME --iterations 10000  --change $CLASSNAME --checkpoint_iterations 10000 --start_checkpoint output/PASLCD/$DATASET_NAME/$CLASSNAME/chkpnt7000.pth --resolution $RESOLUTION 

        echo "Rendering viewpoints for data augmentation"
        python render_viewpoints.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME  --iterations 10000  --change $CLASSNAME  --resolution $RESOLUTION --aug True

        echo "Creating masks for data augmentation"
        python create_mask.py --t $T --input_folder output/PASLCD/$DATASET_NAME/$CLASSNAME

        echo "Updating the change channels with data augmentation"
        python train_masks.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME --iterations 13000  --change $CLASSNAME --checkpoint_iterations 13000 --start_checkpoint output/PASLCD/$DATASET_NAME/$CLASSNAME/chkpnt10000.pth --resolution $RESOLUTION --augment True 

        echo "Rendering viewpoints for final masks generation"
        python render_viewpoints.py -s data/PASLCD/$DATASET_NAME/$CLASSNAME -m output/PASLCD/$DATASET_NAME/$CLASSNAME  --iterations 13000  --change $CLASSNAME  --resolution $RESOLUTION --mask True
    
        python evaluate.py --gt data/PASLCD/$DATASET_NAME/$CLASSNAME/gt_mask --pred_binary output/PASLCD/$DATASET_NAME/$CLASSNAME/renders/binary_masks/ 
    done
done


