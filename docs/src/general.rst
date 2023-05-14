About this project
======================

ECOMOD modeling support system.

A modeling support program for socio-economic processes. The system derives the necessary optimality conditions for complex economic models, in which agents plan their decisions based on optimal management tasks. We will introduce you to the input and output of the model formulation, the description of the optimal management tasks of agents and coordination of their solutions through interactions, the technological aspects of working with mathematical model formulations.

Let us describe the logic of the agent problem construction as follows:

- Selection of the decision-making party, control and external parameters
- Formulation of optimality conditions for the choice of agent control
- Simplification of the set of optimality conditions
- Combining different agent models through interaction models
- Numerical and scenario analysis of problem solution, forming a set of all conditions

The system we create is an end-to-end LaTeX handler for agent-based models created in the Python language.



Relevance
===================

The authors of the "Ecomod" system have repeatedly noted the peculiarities of modeling complex systems, not necessarily in economics. Systems capable of self-development (for example, as a living

organism; biosphere; society: technology, economy, language, culture), in modeling show similar problems. Particular models, which are not particular cases of the universal "supermodel".

partial models, which are not particular cases of the universal "supermodel", describe individual aspects of the investigated system. At the same time, due to the preferences of authors or customers, these constructions operate with different sets of concepts and, being simplified descriptions, neglect certain deviations from the regularities described in them [1], [7]. Sometimes, one team of researchers uses different constructs when modeling economic subsystems - bank, international trade, domestic market, with different detailing or, on the contrary, aggregation of indicators [2],[6]. The description of microsystems in an applied model may use a large number of indicators, which are related by several relations [9], which makes the task of analyzing the model difficult in terms of calculations.

A number of attempts have been made to standardize models in order to be able to combine developments from different models and adapt them to different implementation environments. To work with models of complex systems, a canonical form and a tool system "Ecomod" have been proposed[1]. The canonical form - formalization of model in Ecomod system is not bound to the automation tool and is a unique development of the team. Structural and classification characteristics of the canonical form in ECOMOD system imply special designation of variables, their indexing, and special entries in the headings of groups of relations. The implementation of the system in Maple computer algebra allowed the use of list attributes so that ratio entries and group headings could be supplemented with textual comments. The model presented in Maple's Ecomod system looked user-friendly, describing the model in detail block by block and demonstrating a symbolic notation of relationships. The Ecomod approach allows 6 levels of checking the correctness of the model record in canonical form. Control of levels 1-5 is formalized in the form of checking axioms of the canonical form construction [3].

The article [8] proposes the use of so-called object-oriented ontologies for designing multi-agent modeling systems in microeconomics. By integration, the authors mean the combination of heterogeneous components for large-scale experiments with the model, but they note the importance of formalization for subsequent reuse of the same part of the model in other models or modeling environments. In the paper, the authors propose an approach based on a microeconomics ontology and the key parameters of a modeling tool to implement it bilaterally. The tool allows generating code for a distributed agent-based simulation environment.

The results obtained are focused on the development of simulation models of the economy and are an important study of the issue of model computability.

Team
=================


Zhukova A.A.
-----------------
Project head

FRC CSC RAS, Russia
(Federal research center "Computer science and control" Russian academy of sciences)


Pilnik N.P.
-----------------
Economist

HSE, Russia
(Higher School of Economics)


Kamenev I.G.
-------------
Economist, research consultant

FRC CSC RAS, Russia
(Federal research center "Computer science and control" Russian academy of sciences)


Iusup-Akhunov B.B
-------------------
Python developer, System analyst

MIPT, Russia
(Moscow Institute of Physics and Technology)


References
======================

[1] Петров, А. А., Поспелов, И. Г., Поспелова, Л. Я., Хохлов, М. А. ЭКОМОД-
Интеллектуальный инструмент разработки и исследования динамических моделей эконо-
мики //Материалы II Всероссийской конференции (ИММОД–2005). С.–Петербург. —- 2005.
—- С. 19–21.


[2] Андреев, М. Ю., Вржещ, В. П., Пильник, Н. П., Поспелов, И. Г., Хохлов, М. А., Жукова,
А. А., Радионов, С. А. Модель межвременного равновесия экономики России, основанная
на дезагрегировании макроэкономического баланса //Труды семинара имени ИГ Петров-
ского. —- 2013. —- Т. 29. –– №. 0. —- С. 43–145.


[3] Pospelov I. G., Khokhlov M. A. Dimensionality control method for economy dynamics models
//Matematicheskoe modelirovanie. – 2006. – Т. 18. – №. 10. – С. 113-122.


[4] Лайонс Р. Цифровая обработка сигналов. — М.: Бином, 2006. — С. 361–369.


[5] Khokhlov M. A., Pospelov I. G., Pospelova L. Y. Technology of development and implementation
of realistic (country–specific) models of intertemporal equilibrium //International Journal of
Computational Economics and Econometrics. – 2014. – Т. 4. – №. 1-2. – С. 234-253.


[6] Radionov S., Pilnik N., Pospelov I. The Relaxation of Complementary Slackness Conditions as
a Regularization Method for Optimal Control Problems //Advances in Systems Science and
Applications. – 2019. – Т. 19. – №. 2. – С. 44-62.


[7] L. Y. Pospelova, L. Pospelov, A. Petrov A. ECOMOD - A Modeling Support System for
Mathematical Models of Economy. Proc. Of Intl. Conf. Computing in economy and finance.
Geneva, Switzerland, 1996.


[8] Babkin E., Abdulrab H., Kozyrev O. Application of ontology-based tools for design of multi-
agent simulation environments in economics //Proceedings of the IASTED Asian Conference
on Modelling and Simulation. – 2007. – С. 55-60.


[9] Shevchenko V. On the Construction and Analysis of Macroeconomic Operating Game
Models //2018 Eleventh International Conference"Management of large-scale system
development"(MLSD. – IEEE, 2018. – С. 1-5.