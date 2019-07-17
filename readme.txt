1. Download the given dataset
2. Run train.py --stage "Shape" # This will train Stage 1 Poly-Gan to change structure of Reference cloth
3. Run train.py --stage "Stitch" # This will train Stage 2 Poly-Gan to stitch clothes to missing portion as shown in paper
4. Run train.py --stage "Refine" # This will train Stage 3 Poly-Gan to refine for missing regions in result of Stage 2
5. Run test.py --stage "Refine" # This will give the complete result from Stage 1 - Stage 4.

test.py requires location of pre-trained models for Stage 1, 2 , 3.

for more information on usage please conatct any of the authors on paper.

Thank you.