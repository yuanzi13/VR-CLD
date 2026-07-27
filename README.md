# VR-CLD
A virtual reality-based cognitive load classification framework for older adults
using eye-tracking signals.

Due to ethical restrictions, raw participant data cannot be publicly released. 
A synthetic example dataset is provided to demonstrate the data format and 
preprocessing pipeline.
## License

This project is released under the MIT License.

Copyright (c) 2026 yuanzi13

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Repository Structure

- `data`: example datasets
- `preprocessing/`: preprocessing scripts
- `experiments/binary/`: binary classification experiments
- `experiments/ternary/`: ternary classification experiments
- `config.yaml`: model, training, and preprocessing parameters

## Environment

Python 3.8.20  
PyTorch 2.4.1  
CUDA 11.8

## Example Usage

```bash
python preprocessing/preprocess.py
python experiments/binary/subject_dependent_5fold/tcn.py
