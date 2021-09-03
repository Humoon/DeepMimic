import os
import pickle
from train.common.utils import create_path
from model.network_surgery import Surgeon

old_checkpoint_path = "./checkpoints"
new_checkpoint_path = "./new_checkpoints"

sg = Surgeon(old_checkpoint_path, new_checkpoint_path)

sg.model_input_surgery()
sg.save_new_model()
print("checkpoint surgery done!")
# test your surgery result, see the output difference.
sg.surgery_test(old_share_feature_shape=111, old_player_feature_shape=54)

# model pool surgery
if os.path.exists("./log/model_pools") is True:
    files = os.listdir("./log/model_pools")
    create_path("./log/new_model_pools")
    for i, file in enumerate(files):
        if "model_f" in file:
            with open(f"./log/model_pools/{file}", "rb") as f:
                model_dict = pickle.loads(f.read())
            sg = Surgeon(old_model_dict=model_dict, zero_padding=False)
            new_model_dict = sg.model_input_surgery()
            new_model_dict["update_times"] = model_dict["update_times"]
            new_model_dict["model_time"] = model_dict["model_time"]
            new_model_dict["elo"] = model_dict["elo"]
            with open(f"./log/new_model_pools/{file}", "wb") as f:
                f.write(pickle.dumps(new_model_dict))
            print(i, file, "surgery done.")

# first training server gpu machine run:

# docker run --rm -it \
# -v $(pwd)/newton:/newton -w /newton \
# -v $(pwd)/log:/newton/log \
# -v $(pwd)/checkpoints:/newton/checkpoints \
# football_game python3 train/run/run_surgery.py
