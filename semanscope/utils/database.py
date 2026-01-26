#!/usr/bin/env python3
"""
Database utilities for Semanscope centralized SQLite database management
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanScopeDB:
    """Central database manager for Semanscope datasets"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default path
            self.db_path = Path(__file__).parent.parent.parent / "data" / "semanscope.sqlite"
        else:
            self.db_path = Path(db_path)

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database with core tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create metadata table for tracking datasets
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_name TEXT UNIQUE NOT NULL,
                        source_file TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        row_count INTEGER,
                        column_count INTEGER,
                        version TEXT
                    )
                """)

                # Create NSM Prime Words table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS nsm_prime_words (
                        id INTEGER PRIMARY KEY,
                        word TEXT NOT NULL,
                        domain TEXT,
                        type TEXT,
                        tier INTEGER,
                        language TEXT,
                        nsm_prime_ref TEXT,
                        nsm_prime_group TEXT,
                        position INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsm_tier ON nsm_prime_words(tier)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsm_language ON nsm_prime_words(language)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsm_domain ON nsm_prime_words(domain)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsm_group ON nsm_prime_words(nsm_prime_group)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsm_word ON nsm_prime_words(word)")

                conn.commit()
                logger.info(f"Database initialized at: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def load_csv_to_table(self, csv_path: str, table_name: str, description: str = "") -> bool:
        """Load CSV data into specified table"""
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False

            # Read CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from {csv_path}")

            # Load into database
            with self.get_connection() as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)

                # Update metadata
                self._update_metadata(conn, table_name, str(csv_path), description, len(df), len(df.columns))

            logger.info(f"Successfully loaded data into table: {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading CSV to table: {e}")
            return False

    def _update_metadata(self, conn, table_name: str, source_file: str, description: str,
                        row_count: int, column_count: int):
        """Update dataset metadata"""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO dataset_metadata
            (table_name, source_file, description, row_count, column_count, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (table_name, source_file, description, row_count, column_count))
        conn.commit()

    def query_nsm_words(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Query NSM Prime Words with optional filters"""
        base_query = "SELECT * FROM nsm_prime_words"
        params = []
        conditions = []

        if filters:
            if 'domain' in filters and filters['domain'] != 'All':
                conditions.append("domain = ?")
                params.append(filters['domain'])

            if 'type' in filters and filters['type'] != 'All':
                conditions.append("type = ?")
                params.append(filters['type'])

            if 'language' in filters and filters['language'] != 'All':
                conditions.append("language = ?")
                params.append(filters['language'])

            if 'tier' in filters and filters['tier'] != 'All':
                conditions.append("tier = ?")
                params.append(filters['tier'])

            if 'nsm_prime_group' in filters and filters['nsm_prime_group'] != 'All':
                conditions.append("nsm_prime_group = ?")
                params.append(filters['nsm_prime_group'])

            if 'word_search' in filters and filters['word_search']:
                conditions.append("word LIKE ?")
                params.append(f"%{filters['word_search']}%")

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY id ASC"

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(base_query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error querying NSM words: {e}")
            return pd.DataFrame()

    def get_filter_options(self) -> Dict[str, List[str]]:
        """Get unique values for filter dropdowns"""
        try:
            with self.get_connection() as conn:
                options = {}

                # Get unique domains
                df = pd.read_sql_query("SELECT DISTINCT domain FROM nsm_prime_words WHERE domain IS NOT NULL ORDER BY domain", conn)
                options['domains'] = ['All'] + df['domain'].tolist()

                # Get unique types
                df = pd.read_sql_query("SELECT DISTINCT type FROM nsm_prime_words WHERE type IS NOT NULL ORDER BY type", conn)
                options['types'] = ['All'] + df['type'].tolist()

                # Get unique languages
                df = pd.read_sql_query("SELECT DISTINCT language FROM nsm_prime_words WHERE language IS NOT NULL ORDER BY language", conn)
                options['languages'] = ['All'] + df['language'].tolist()

                # Get unique tiers
                df = pd.read_sql_query("SELECT DISTINCT tier FROM nsm_prime_words WHERE tier IS NOT NULL ORDER BY tier", conn)
                options['tiers'] = ['All'] + df['tier'].tolist()

                # Get unique NSM groups
                df = pd.read_sql_query("SELECT DISTINCT nsm_prime_group FROM nsm_prime_words WHERE nsm_prime_group IS NOT NULL ORDER BY nsm_prime_group", conn)
                options['nsm_groups'] = ['All'] + df['nsm_prime_group'].tolist()

                return options

        except Exception as e:
            logger.error(f"Error getting filter options: {e}")
            return {
                'domains': ['All'],
                'types': ['All'],
                'languages': ['All'],
                'tiers': ['All'],
                'nsm_groups': ['All']
            }

    def get_dataset_stats(self) -> Dict[str, int]:
        """Get basic statistics about the dataset"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Total words
                cursor.execute("SELECT COUNT(*) FROM nsm_prime_words")
                stats['total_words'] = cursor.fetchone()[0]

                # Unique domains
                cursor.execute("SELECT COUNT(DISTINCT domain) FROM nsm_prime_words WHERE domain IS NOT NULL")
                stats['unique_domains'] = cursor.fetchone()[0]

                # Unique languages
                cursor.execute("SELECT COUNT(DISTINCT language) FROM nsm_prime_words WHERE language IS NOT NULL")
                stats['languages'] = cursor.fetchone()[0]

                # Unique NSM groups
                cursor.execute("SELECT COUNT(DISTINCT nsm_prime_group) FROM nsm_prime_words WHERE nsm_prime_group IS NOT NULL")
                stats['nsm_groups'] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            logger.error(f"Error getting dataset stats: {e}")
            return {
                'total_words': 0,
                'unique_domains': 0,
                'languages': 0,
                'nsm_groups': 0
            }

# Global database instance
_db_instance = None

def get_database() -> SemanScopeDB:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SemanScopeDB()
    return _db_instance

def initialize_nsm_data():
    """Initialize NSM Prime Words data from CSV"""
    db = get_database()
    csv_path = Path(__file__).parent.parent.parent / "data" / "input" / "ICML-NSM-Prime-Words.csv"

    if csv_path.exists():
        success = db.load_csv_to_table(
            str(csv_path),
            "nsm_prime_words",
            "ICML NSM Prime Words dataset with multilingual translations"
        )
        if success:
            logger.info("NSM Prime Words data initialized successfully")
        else:
            logger.error("Failed to initialize NSM Prime Words data")
    else:
        logger.warning(f"NSM CSV file not found: {csv_path}")

if __name__ == "__main__":
    # Test the database
    initialize_nsm_data()
    db = get_database()
    stats = db.get_dataset_stats()
    print(f"Database stats: {stats}")