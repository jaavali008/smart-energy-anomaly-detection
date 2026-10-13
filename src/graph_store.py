from neo4j import GraphDatabase

LOAD = """
MERGE (hs:House {id: 'H1'})
MERGE (m:Meter {id: 'M1'})
MERGE (hs)-[:HAS_METER]->(m)
WITH m
UNWIND $rows AS r
CREATE (a:Anomaly {timestamp: r.timestamp, power_kw: r.power_kw,
                   severity: r.severity, excess_kwh: r.excess_kwh,
                   co2_kg: r.co2_kg})
MERGE (s:Source {name: r.main_source})
CREATE (m)-[:HAD_ANOMALY]->(a)
CREATE (a)-[:CAUSED_BY]->(s)
"""

Q_SOURCES = """
MATCH (a:Anomaly)-[:CAUSED_BY]->(s:Source)
RETURN s.name AS source, count(a) AS anomalies, round(sum(a.excess_kwh),1) AS excess_kwh
ORDER BY anomalies DESC
"""

Q_HIGH = """
MATCH (a:Anomaly)-[:CAUSED_BY]->(s:Source)
WHERE a.severity = 'High'
RETURN s.name AS source, count(a) AS high_alerts
ORDER BY high_alerts DESC
"""


class GraphStore:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j")

    def load(self, alert_df):
        rows = alert_df.to_dict("records")
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")      # avoid duplicates on re-run
            s.run(LOAD, rows=rows)

    def report(self):
        with self.driver.session() as s:
            print("\nAnomalies per source:")
            for r in s.run(Q_SOURCES):
                print(dict(r))
            print("\nHigh severity alerts per source:")
            for r in s.run(Q_HIGH):
                print(dict(r))

    def close(self):
        self.driver.close()