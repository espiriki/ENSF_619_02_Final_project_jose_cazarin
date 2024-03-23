# How to repro CHAT GPT Results"

Use the script `chat_GPT_results.py`

You will need to add your OpenAI API key in the variable
`OPEN_AI_API_KEY` defined at the beginning of the script.

The variable `MODEL` is used to set which ChatGPT model will be used. In the paper 3 models were used: `gpt-4-0125-preview`, `gpt-3.5-turbo-0125` and `gpt-4-0613`. Just leave the one you want to use
uncommented.

You will need to adjust the path to the dataset in the
`PATH_TO_DATASET` variable.

To prevent hallucinations in the model, this script only sends 10 objetcs at a time.

The script takes as arguments the ground truth class of the objets plus the start idx of the 10-object chunk that will be sent to the model.

Usage example:

`chat_GPT_results.py 0 Black`
`chat_GPT_results.py 1 Black`
`chat_GPT_results.py 2 Black`
`chat_GPT_results.py 5 Green`

An example of another script that can be used to call this script is:

    classes = ["Black", "Blue", "Green", "TTR"]

    for _class in classes:
        path = os.path.join(PATH_TO_DATASET, _class)
        samples_for_current_class = [f for f in listdir(path) if isfile(join(path, f))]
        num_commands = int(len(samples_for_current_class)/10) + 1
        for i in range(num_commands):
            subprocess.run(["python3", "test_set_chat_GPT.py", str(i), str(_class)])

The above snippet takes all samples for a given class, calculates the number of commands needed to send all samples in chunks of 10, and calls the `chat_GPT_results.py` script accordingly.