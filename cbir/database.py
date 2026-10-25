import sqlite3
import json
import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id TEXT UNIQUE NOT NULL,
                        file_path TEXT NOT NULL,
                        dataset TEXT NOT NULL,
                        category TEXT,
                        filename TEXT,
                        width INTEGER,
                        height INTEGER,
                        file_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS descriptors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id TEXT NOT NULL REFERENCES images(image_id),
                        descriptor_type TEXT NOT NULL,
                        cache_path TEXT NOT NULL,
                        dimensions INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(image_id, descriptor_type)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS index_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        index_name TEXT UNIQUE,
                        descriptor_type TEXT,
                        distance_metric TEXT,
                        num_items INTEGER,
                        build_time_seconds REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS indexing_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset TEXT,
                        status TEXT DEFAULT 'pending',
                        total_images INTEGER DEFAULT 0,
                        processed_images INTEGER DEFAULT 0,
                        errors INTEGER DEFAULT 0,
                        error_log TEXT,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                ''')
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def add_image(self, image_id, file_path, dataset, category=None, filename=None, width=None, height=None, file_size=None) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO images 
                    (image_id, file_path, dataset, category, filename, width, height, file_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (image_id, file_path, dataset, category, filename, width, height, file_size))
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding image {image_id}: {e}")
            return False

    def get_image(self, image_id) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_images_by_dataset(self, dataset, limit=None, offset=None) -> List[Dict]:
        query = "SELECT * FROM images WHERE dataset = ?"
        params = [dataset]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)
            
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_category(self, category) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM images WHERE category = ?", (category,))
            return [dict(row) for row in cursor.fetchall()]

    def search_images_by_text(self, query: str) -> List[Dict]:
        search_term = f"%{query}%"
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM images 
                WHERE category LIKE ? OR filename LIKE ? OR image_id LIKE ?
            ''', (search_term, search_term, search_term))
            return [dict(row) for row in cursor.fetchall()]

    def add_descriptor(self, image_id, descriptor_type, cache_path, dimensions) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO descriptors 
                    (image_id, descriptor_type, cache_path, dimensions)
                    VALUES (?, ?, ?, ?)
                ''', (image_id, descriptor_type, cache_path, dimensions))
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding descriptor for image {image_id}: {e}")
            return False

    def get_descriptor_path(self, image_id, descriptor_type) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT cache_path FROM descriptors 
                WHERE image_id = ? AND descriptor_type = ?
            ''', (image_id, descriptor_type))
            row = cursor.fetchone()
            return row['cache_path'] if row else None

    def get_images_with_descriptor(self, descriptor_type, dataset=None) -> List[Dict]:
        query = '''
            SELECT i.*, d.cache_path, d.dimensions 
            FROM images i 
            JOIN descriptors d ON i.image_id = d.image_id 
            WHERE d.descriptor_type = ?
        '''
        params = [descriptor_type]
        if dataset:
            query += " AND i.dataset = ?"
            params.append(dataset)
            
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_dataset_stats(self, dataset=None) -> Dict:
        stats = {}
        with self._get_connection() as conn:
            # Total images
            query = "SELECT COUNT(*) as count FROM images"
            params = []
            if dataset:
                query += " WHERE dataset = ?"
                params.append(dataset)
            cursor = conn.execute(query, params)
            stats['total_images'] = cursor.fetchone()['count']
            
            # Categories
            query = "SELECT category, COUNT(*) as count FROM images"
            params = []
            if dataset:
                query += " WHERE dataset = ?"
                params.append(dataset)
            query += " GROUP BY category"
            cursor = conn.execute(query, params)
            stats['categories'] = {row['category']: row['count'] for row in cursor.fetchall()}
            
            # Descriptor coverage
            query = '''
                SELECT d.descriptor_type, COUNT(DISTINCT d.image_id) as count
                FROM descriptors d
                JOIN images i ON d.image_id = i.image_id
            '''
            params = []
            if dataset:
                query += " WHERE i.dataset = ?"
                params.append(dataset)
            query += " GROUP BY d.descriptor_type"
            cursor = conn.execute(query, params)
            stats['descriptors'] = {row['descriptor_type']: row['count'] for row in cursor.fetchall()}
            
        return stats

    def update_indexing_status(self, dataset, **kwargs):
        allowed_fields = {'status', 'total_images', 'processed_images', 'errors', 'error_log', 'started_at', 'completed_at'}
        updates = []
        params = []
        
        for k, v in kwargs.items():
            if k in allowed_fields:
                updates.append(f"{k} = ?")
                params.append(v)
                
        if not updates:
            return
            
        params.append(dataset)
        query = f"UPDATE indexing_status SET {', '.join(updates)} WHERE dataset = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT id FROM indexing_status WHERE dataset = ?", (dataset,))
                if cursor.fetchone() is None:
                    conn.execute("INSERT INTO indexing_status (dataset) VALUES (?)", (dataset,))
                conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"Error updating indexing status for {dataset}: {e}")

    def get_indexing_status(self, dataset=None) -> Dict:
        with self._get_connection() as conn:
            if dataset:
                cursor = conn.execute("SELECT * FROM indexing_status WHERE dataset = ?", (dataset,))
                row = cursor.fetchone()
                return dict(row) if row else {}
            else:
                cursor = conn.execute("SELECT * FROM indexing_status")
                return {row['dataset']: dict(row) for row in cursor.fetchall()}

    def add_index_metadata(self, index_name, descriptor_type, distance_metric, num_items, build_time):
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO index_metadata 
                    (index_name, descriptor_type, distance_metric, num_items, build_time_seconds)
                    VALUES (?, ?, ?, ?, ?)
                ''', (index_name, descriptor_type, distance_metric, num_items, build_time))
        except sqlite3.Error as e:
            logger.error(f"Error adding index metadata: {e}")

    def get_all_categories(self, dataset=None) -> List[str]:
        query = "SELECT DISTINCT category FROM images"
        params = []
        if dataset:
            query += " WHERE dataset = ?"
            params.append(dataset)
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [row['category'] for row in cursor.fetchall() if row['category']]

    def image_exists(self, image_id) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM images WHERE image_id = ?", (image_id,))
            return cursor.fetchone() is not None

    def descriptor_exists(self, image_id, descriptor_type) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM descriptors WHERE image_id = ? AND descriptor_type = ?", (image_id, descriptor_type))
            return cursor.fetchone() is not None

    def get_total_images(self, dataset=None) -> int:
        query = "SELECT COUNT(*) as count FROM images"
        params = []
        if dataset:
            query += " WHERE dataset = ?"
            params.append(dataset)
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()['count']

    def close(self):
        # connections are created on the fly with context managers, so no persistent connection to close
        pass
