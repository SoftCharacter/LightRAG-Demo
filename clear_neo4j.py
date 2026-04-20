"""
Script to clear Neo4j database
"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "12345678")

print(f"Connecting to Neo4j at {uri}...")
driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        # Delete all nodes and relationships
        result = session.run("MATCH (n) DETACH DELETE n")
        print("✅ All nodes and relationships deleted successfully")
        
        # Count remaining nodes
        count_result = session.run("MATCH (n) RETURN count(n) as count")
        count = count_result.single()["count"]
        print(f"Remaining nodes: {count}")
        
finally:
    driver.close()
    print("Neo4j connection closed")

