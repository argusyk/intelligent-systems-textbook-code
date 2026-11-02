# Ontology of APU faults in RDF and SPARQL queries (rdflib library)
from rdflib import Graph

TTL = """
@prefix apu:  <http://example.org/apu#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

apu:FuelSystemFault  rdfs:subClassOf apu:Fault .
apu:StartSystemFault rdfs:subClassOf apu:Fault .
apu:SensorFault      rdfs:subClassOf apu:Fault .
apu:DirtyNozzles     rdfs:subClassOf apu:FuelSystemFault ; rdfs:label "dirty nozzles" ;
    apu:hasSymptom apu:HighEGT, apu:SlowStart, apu:HighFuelFlow ; apu:affects apu:Nozzles .
apu:FuelFilterClog   rdfs:subClassOf apu:FuelSystemFault ;
    rdfs:label "clogged fuel filter" ;
    apu:hasSymptom apu:SlowStart, apu:LowFuelPressure ; apu:affects apu:FuelFilter .
apu:WeakStarter      rdfs:subClassOf apu:StartSystemFault ; rdfs:label "weak starter" ;
    apu:hasSymptom apu:SlowStart ; apu:affects apu:Starter .
apu:BadThermocouple  rdfs:subClassOf apu:SensorFault ; rdfs:label "faulty thermocouple" ;
    apu:hasSymptom apu:HighEGT ; apu:affects apu:Thermocouple .
apu:Nozzles apu:checkHours 4 .  apu:FuelFilter apu:checkHours 1 .
apu:Starter apu:checkHours 2 .  apu:Thermocouple apu:checkHours 0.5 .
"""
g = Graph().parse(data=TTL, format="turtle")
print("triples in the graph:", len(g))

Q1 = """PREFIX apu: <http://example.org/apu#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name WHERE { ?f rdfs:subClassOf+ apu:FuelSystemFault ; rdfs:label ?name . }"""
print("fuel system faults:", [str(r.name) for r in g.query(Q1)])

Q2 = """PREFIX apu: <http://example.org/apu#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name (COUNT(?s) AS ?n) ?h WHERE {
  VALUES ?s { apu:HighEGT apu:SlowStart }
  ?f apu:hasSymptom ?s ; rdfs:label ?name ; apu:affects ?c . ?c apu:checkHours ?h .
} GROUP BY ?name ?h ORDER BY DESC(?n) ?h"""
print("hypotheses for 'overheat + prolonged start' (symptom matches, hours to check):")
for r in g.query(Q2):
    print(f"  {r.name}: {r.n}, {r.h}")
