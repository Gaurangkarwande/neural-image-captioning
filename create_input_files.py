from utils import create_input_files

data_folder = '/home/gaurangajitk/DL/data/image-caption-data'

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(data_folder, 
                        captions_per_image=5,
                        min_word_freq=15,
                        output_folder=data_folder,
                        max_len=50)