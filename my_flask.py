
from flask import Flask, render_template, request, make_response
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from gensim.models import KeyedVectors
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO

model = KeyedVectors.load_word2vec_format(
    'modelnew.model', binary=False)


# class MyHandler(FileSystemEventHandler):
#     def on_modified(self, event):
#         app.logger.info('Reloading due to change in %s', event.src_path)
#         app.logger.info('Restarting...')
#         os.kill(os.getpid(), signal.SIGTERM)


app = Flask(__name__)

# define route for home page


@app.route('/', methods=['GET', 'POST'])
def home():
    # initialize user input variable
    user_input = None

    # check if request method is POST
    if request.method == 'POST':
    # get user input from form
        user_input = request.form.get('sentence')
        punctuation_pattern = re.compile(r'[^\w\s]')

        embeddings = []
        oov_words = []
        user_input_cleaned = punctuation_pattern.sub(
            '', user_input)  # remove all punctuation marks
        for word in user_input_cleaned.split():
            try:
                embedding = model[word]
                embeddings.append(embedding)
            except KeyError:
                error_message = "The words you input are out of vocabulary. Enter words again"
                return render_template('index.html', error_message=error_message)

        word_embeddings_3d = np.array(embeddings)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            word_embeddings_3d[:, 0], word_embeddings_3d[:, 1], word_embeddings_3d[:, 2])
        for i, word in enumerate(user_input_cleaned.split()):
            ax.text(
                word_embeddings_3d[i, 0], word_embeddings_3d[i, 1], word_embeddings_3d[i, 2], word)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Return the image data as a response with the correct content type
        response = make_response(img_buffer.getvalue())
        response.mimetype = 'image/png'
        return response

# render HTML template
    return render_template('index.html')



if __name__ == '__main__':
    # observer = Observer()
    # observer.schedule(MyHandler(), '.', recursive=True)
    # observer.start()
    app.run(debug=True)
