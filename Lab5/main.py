from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

game_model = BayesianNetwork(
    [
        ("Carte1", "Jucator1"),
        ("Carte2", "Jucator2"),
        ("Jucator1", "D1"),
        ("Jucator2", "D2"),
        ("D1","D2"),
        ("D2","D3"),
        ("D3","Joc"),
        ("D2","Joc")




    ]
)

cpd_CarteJ1 = TabularCPD("Carte1",5,[[0.20],[0.20],[0.20],[0.20],[0.20]])
cpd_CarteJ2 = TabularCPD(variable="Carte2",
                         variable_card=5,
                         values=[[0.0,0.25,0.25,0.25,0.25],
                                 [0.25,0.0,0.25,0.25,0.25],
                                 [0.25,0.25,0.0,0.25,0.25],
                                 [0.25,0.25,0.25,0.0,0.25],
                                 [0.25,0.25,0.25,0.25,0.0]],
                         evidence=["Carte1"],
                         evidence_card=[5]
                         )
cpd_Jucator1= TabularCPD(variable="Jucator1",
                         variable_card=2,
                         values=[
                             [0,0.20,0.60,0.80,1],
                             [1,0.8,0.4,0.2,0],
                         ],
                         evidence=["Carte1"],
                         evidence_card=[5],
                        )
cpd_Jucator2=TabularCPD(variable="Jucator2",
                         variable_card=2,
                         values=[
                             [0,0.10,0.40,0.80,1],
                             [1,0.9,0.6,0.2,0]
                         ],
                         evidence=["Carte2"],
                         evidence_card=[5],
                        )

cpd_D1=TabularCPD(variable="D1",
                         variable_card=2,
                         values=[
                             [0,0.20,0.60,0.80,1],
                             [1,0.8,0.4,0.2,0],
                         ],
                         evidence=["Carte1"],
                         evidence_card=[5],
                  )

cpd_D2=TabularCPD(variable="D2",
                         variable_card=2,
                         values=[
                             [0,0.10,0.40,0.80,1],
                             [1,0.9,0.6,0.2,0]
                         ],
                         evidence=["Carte2"],
                         evidence_card=[5],
                        )
cpd_D3=TabularCPD(variable="Joc",
                  variable_card=2,
                  values=[],
                  evidence_card=["D1","D2"],
                  evidence_card=[]
                  )






