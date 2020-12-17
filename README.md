A deployment for DRFI model. Choose your image and it will help you to crop the most salient object in scene.

1. Prerequisite

- Create an environment with virtualenv:

    ```bash
    virtualenv drfi-flask -p python3
    ```

    This command will create a virtual enviroment with name drfi-flask and python3 for default python.

- Activate your environment:

    ```bash
    source drfi-flask/bin/activate
    ```

- Install all libraries with pip:

    ```bash
    pip install -r requirements.txt
    ```

2.  Load your trained model:

- Download pretrained model, copy it to drfi/data/model folder.

3.  Run your server:

Notice: You need to activate your environment before start your server.

```bash
python server.py
```

or

```bash
flask run
```

3. Run an inference: Hey, just choose your image, wait for a moment, and get result. That's it.
