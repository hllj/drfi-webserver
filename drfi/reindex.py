import os

ROOT_FOLDER = os.getcwd()
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data/MSRA-B/")

files = os.listdir(DATA_FOLDER)
if __name__ == "__main__":
    files_name = set()
    for file in files:
        file_name = file[:-4]
        files_name.add(file_name)
    for index, file_name in enumerate(files_name):
        print("Reindex for {} with index {}".format(file_name, index))
        os.rename(
            os.path.join(
                DATA_FOLDER,
                "{}.jpg".format(file_name),
            ),
            os.path.join(DATA_FOLDER, "{}.jpg".format(str(index))),
        )

        os.rename(
            os.path.join(
                DATA_FOLDER,
                "{}.png".format(file_name),
            ),
            os.path.join(DATA_FOLDER, "{}.png".format(str(index))),
        )
    print("Reindex completed")
