# chatbot_nltk
Chatbot for Treefrog  Consulting built with Python and NLTK

# Notes
I have run this program in a Docker container due to tensorflow not being compatible with the latest version of Python.
The same result could also be achieve using a virtual enivironment, or if your machine is running Python 3.6.x
The model is trained using the intents.json file, which can be switched out to train a bot with a different purpose.
The model is created and trained once to save computing resources but if you delete the data.pickle file and the saved model files it will create and train a new model.

# How to run
The bot is run by simply entering *Python main.py* into the command lineIif this is the first run the bot will be created and trained, you can increase or decrease the amount of training done by adjusting the epoch number in the model.fit call
Once the training is completed the use will be prompted to communicate with the bot
Once the user is finished talking to the bot, they can press Q or q to exit the program

### Commands to train and start a conversation
```bash
python -m pip install -r requirements.txt
python -m main.py
```