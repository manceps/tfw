import flask
import bernie

app = flask.Flask(__name__, static_url_path='/static/')

MODEL_FILENAME = 'models/bern.iter999.h5'
TEXT_FILENAME = 'bernie_corpus.txt'

import train_bernie
text = train_bernie.read_text_from_file(TEXT_FILENAME)
char_indices, indices_char = bernie.make_char_lookup_table(text)
model = bernie.load_model(char_indices)

"""
@app.route('/visualization')
def visualization():
    weights = open('/tmp/visualization.png').read()
    return flask.Response(weights, mimetype='image/png')
"""

@app.route('/ask_question')
def ask_question():
    question = flask.request.args.get('question') + '. '
    print('Received question: {}'.format(question))
    answer = bernie.ask_bernie(model, question, char_indices, indices_char)
    print('Returning answer: {}'.format(answer))
    return answer

@app.route('/')
def route_index():
    return app.send_static_file('bernie.html')

@app.route('/<path:path>')
def send_static_file(path):
    return flask.send_from_directory('static', path)

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, use_reloader=False)
