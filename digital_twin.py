import torch
from model import Generator
from flask import Flask, request, jsonify

app = Flask(__name__)

generator = Generator()
generator.load_state_dict(torch.load("global_generator.pth"))
generator.eval()

@app.route('/generate', methods=['POST'])
def generate_image():
    noise = torch.randn(1, 100)
    generated_image = generator(noise).detach().cpu().numpy().tolist()
    return jsonify({"generated_image": generated_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
