# quantumdojo
![Quantum Intuition Dojo](qid.png)
###### _Image provided by [@dncolomer](https://github.com/dncolomer) via [Canva](https://about.canva.com/license-agreements/)_

A Quantum Computing Dojo for improving your intuition inspired by [IBM's May 4th, 2020 Quantum Challenge](https://github.com/qiskit-community/may4_challenge) and from discussion with [@dncolomer](https://github.com/dncolomer), creator of the YouTube channel _[Quantum Intuition](https://www.youtube.com/channel/UC-2knDbf4kzT3uzWo7iTJyw)_.

> "A dōjō is a hall or place for immersive learning or meditation. This is traditionally in the field of martial arts, but has been seen increasingly in other fields, such as meditation and software development. The term literally means "place of the Way" in Japanese." -Wikipedia

## To run
_NOTE: Requires [qiskit installation](https://qiskit.org/documentation/install.html)_   
`jupyter notebook Dojo.ipynb`

## To compete or for minimal mode
`jupyter notebook Compete.ipynb`

## As a developer
1. Run Dojo.ipynb and additionally Sensei.ipynb
2. Develop locally in Sensei.ipynb if need be 
3. Export to Sensei.py (make sure to remove any main execution code!):
   `jupyter nbconvert --to script Sensei.ipynb`
