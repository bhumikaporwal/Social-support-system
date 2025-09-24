from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pymongo import MongoClient
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from typing import Generator
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# Database Connection - Try PostgreSQL first, fallback to SQLite
try:
    # Try PostgreSQL connection
    engine = create_engine(
        settings.postgres_url,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.environment == "development"
    )
    # Test the connection
    engine.connect().close()
    logger.info("Connected to PostgreSQL database")
except Exception as e:
    # Fallback to SQLite
    logger.warning(f"PostgreSQL connection failed ({e}), falling back to SQLite")
    engine = create_engine(
        settings.sqlite_url,
        echo=settings.environment == "development"
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# MongoDB Connection
class MongoDBClient:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._client = MongoClient(settings.mongodb_url)
            self.db = self._client[settings.mongodb_db]

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def close(self):
        if self._client:
            self._client.close()

# Qdrant Connection
class QdrantConnection:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )

    @property
    def client(self):
        return self._client

# Neo4j Connection
class Neo4jConnection:
    _instance = None
    _driver = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )

    def get_session(self):
        return self._driver.session()

    def close(self):
        if self._driver:
            self._driver.close()

# Initialize connections (with error handling)
try:
    mongodb_client = MongoDBClient()
except Exception as e:
    logger.warning(f"MongoDB client initialization failed: {e}")
    mongodb_client = None

try:
    qdrant_client = QdrantConnection()
except Exception as e:
    logger.warning(f"Qdrant client initialization failed: {e}")
    qdrant_client = None

try:
    neo4j_client = Neo4jConnection()
except Exception as e:
    logger.warning(f"Neo4j client initialization failed: {e}")
    neo4j_client = None

def init_databases():
    """Initialize all database connections and create tables"""
    try:
        # Create SQL database tables
        from src.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("SQL database tables created successfully")

        # Try to initialize Qdrant collections
        if qdrant_client is not None:
            try:
                collections = ["applications", "documents", "policies"]
                for collection in collections:
                    try:
                        qdrant_client.client.create_collection(
                            collection_name=collection,
                            vectors_config={
                                "size": 768,  # OpenAI embeddings size
                                "distance": "Cosine"
                            }
                        )
                        logger.info(f"Qdrant collection '{collection}' created")
                    except Exception as e:
                        logger.warning(f"Qdrant collection '{collection}' might already exist: {e}")
                logger.info("Qdrant collections initialized")
            except Exception as e:
                logger.warning(f"Qdrant initialization failed: {e}. Vector search will be disabled.")
        else:
            logger.warning("Qdrant client not available. Vector search will be disabled.")

        # Try to test Neo4j connection
        if neo4j_client is not None:
            try:
                with neo4j_client.get_session() as session:
                    result = session.run("RETURN 1")
                    result.single()
                    logger.info("Neo4j connection successful")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Graph features will be disabled.")
        else:
            logger.warning("Neo4j client not available. Graph features will be disabled.")

        logger.info("Database initialization completed (some services may be disabled)")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def close_connections():
    """Close all database connections"""
    if mongodb_client is not None:
        mongodb_client.close()
    if neo4j_client is not None:
        neo4j_client.close()
    logger.info("All database connections closed")