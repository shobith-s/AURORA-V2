#!/usr/bin/env python3
"""
AURORA V2 - Database Viewer for Presentation
Quick script to show corrections stored in SQLite database
"""

import sqlite3
import json
from datetime import datetime
import sys

def view_corrections():
    """Display all corrections in a presentation-friendly format"""
    
    try:
        conn = sqlite3.connect('aurora.db')
        cursor = conn.cursor()
        
        print('='*80)
        print('AURORA V2 - CORRECTIONS DATABASE')
        print('='*80)
        
        # Count total corrections
        cursor.execute('SELECT COUNT(*) FROM corrections')
        total = cursor.fetchone()[0]
        
        print(f'\n📊 TOTAL CORRECTIONS STORED: {total}')
        
        if total == 0:
            print('\n⚠️  No corrections in database yet.')
            print('   Run the adaptive learning demo to add sample corrections.')
            conn.close()
            return
        
        # Show corrections by action
        print('\n📈 BREAKDOWN BY CORRECTED ACTION:')
        print('-'*80)
        cursor.execute('''
            SELECT correct_action, COUNT(*) as count
            FROM corrections
            GROUP BY correct_action
            ORDER BY count DESC
        ''')
        
        for row in cursor.fetchall():
            action = row[0]
            count = row[1]
            bar = '█' * min(count, 50)
            print(f'  {action:<25} {count:>3} {bar}')
        
        # Show all corrections with details
        print(f'\n📝 ALL CORRECTIONS (Showing all {total}):')
        print('='*80)
        
        cursor.execute('''
            SELECT 
                id,
                wrong_action,
                correct_action,
                ROUND(system_confidence, 2) as conf,
                timestamp,
                pattern_hash,
                statistical_fingerprint
            FROM corrections
            ORDER BY timestamp DESC
        ''')
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f'\n[{i}] CORRECTION ID: {row[0]}')
            print(f'    Pattern Hash: {row[5]}')
            print(f'    Wrong Action: {row[1]}')
            print(f'    Correct Action: {row[2]} ✓')
            print(f'    System Confidence: {row[3]}')
            print(f'    Timestamp: {datetime.fromtimestamp(row[4]).strftime("%Y-%m-%d %H:%M:%S")}')
            
            # Show key statistics
            if row[6]:
                stats = json.loads(row[6])
                print(f'    Key Statistics:')
                
                if 'skewness_anonymous' in stats:
                    print(f'      - Skewness: {stats["skewness_anonymous"]:.2f}')
                if 'kurtosis_anonymous' in stats:
                    print(f'      - Kurtosis: {stats["kurtosis_anonymous"]:.2f}')
                if 'cardinality' in stats:
                    print(f'      - Cardinality: {stats["cardinality"]}')
                if 'unique_ratio_bucket' in stats:
                    print(f'      - Unique Ratio Bucket: {stats["unique_ratio_bucket"]}')
                if 'is_numeric' in stats:
                    print(f'      - Is Numeric: {stats["is_numeric"]}')
                if 'is_categorical' in stats:
                    print(f'      - Is Categorical: {stats["is_categorical"]}')
        
        print('\n' + '='*80)
        print('✅ DATABASE VIEW COMPLETE')
        print('='*80)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f'❌ Database Error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'❌ Error: {e}')
        sys.exit(1)

def view_tables():
    """Show all tables in the database"""
    
    try:
        conn = sqlite3.connect('aurora.db')
        cursor = conn.cursor()
        
        print('\n' + '='*80)
        print('DATABASE TABLES')
        print('='*80)
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = cursor.fetchone()[0]
            print(f'\n📊 Table: {table_name}')
            print(f'   Rows: {count}')
            
            # Show schema
            cursor.execute(f'PRAGMA table_info({table_name})')
            columns = cursor.fetchall()
            print(f'   Columns:')
            for col in columns:
                print(f'     - {col[1]} ({col[2]})')
        
        print('\n' + '='*80)
        
        conn.close()
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    print('\n')
    
    # Show tables first
    view_tables()
    
    # Show corrections
    view_corrections()
    
    print('\n💡 TIP: To add sample corrections, run:')
    print('   python3 demo_adaptive_learning.py')
    print('\n')
