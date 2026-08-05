"""电力业务数据查询 — SQLite 封装（企业版：带 tenant_id）"""
import sqlite3
import logging
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("./data/power.db")


class DataTools:
    def __init__(self):
        self.db_path = str(DB_PATH)
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT DEFAULT 'default',
                    device_name TEXT, device_type TEXT, voltage_level TEXT,
                    rated_params TEXT, manufacturer TEXT, install_date TEXT,
                    station_name TEXT, status TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fault_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT DEFAULT 'default',
                    device_id TEXT, fault_type TEXT, fault_desc TEXT,
                    fault_time TEXT, recovery_time TEXT,
                    cause_analysis TEXT, disposal_action TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT DEFAULT 'default',
                    project_name TEXT, voltage_level TEXT, project_type TEXT,
                    total_cost REAL, material_cost REAL, labor_cost REAL,
                    created_at TEXT, source_file TEXT
                )""")
            # 演示数据
            if conn.execute("SELECT COUNT(*) FROM device_ledger").fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO device_ledger VALUES (1,'default','#1主变','变压器','220kV',"
                    "'240MVA','特变电工','2024-01','济南变电站','运行')")
                conn.execute(
                    "INSERT INTO fault_records VALUES (1,'default','dev_001','跳闸',"
                    "'220kV线路A相接地','2024-06-15 10:23:00','2024-06-15 11:05:00',"
                    "'雷击','巡线更换绝缘子')")
                conn.execute(
                    "INSERT INTO cost_data VALUES (1,'default','济南110kV线路工程','110kV',"
                    "'线路',1250.5,800.3,200.1,'2024-05','data/jn_110kv.xlsx')")
            conn.commit()

    def query_equipment(self, device_type: str = None, voltage_level: str = None,
                        tenant_id: str = "default") -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM device_ledger WHERE tenant_id=?"
        params = [tenant_id]
        if device_type:
            sql += " AND device_type = ?"; params.append(device_type)
        if voltage_level:
            sql += " AND voltage_level = ?"; params.append(voltage_level)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_fault_records(self, device_id: str = None, limit: int = 20,
                            tenant_id: str = "default") -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        if device_id:
            rows = conn.execute(
                "SELECT * FROM fault_records WHERE tenant_id=? AND device_id=? "
                "ORDER BY fault_time DESC LIMIT ?", (tenant_id, device_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fault_records WHERE tenant_id=? ORDER BY fault_time DESC LIMIT ?",
                (tenant_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_cost_data(self, voltage_level: str = None, limit: int = 20,
                        tenant_id: str = "default") -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        if voltage_level:
            rows = conn.execute(
                "SELECT * FROM cost_data WHERE tenant_id=? AND voltage_level=? "
                "ORDER BY created_at DESC LIMIT ?", (tenant_id, voltage_level, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cost_data WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


data_tools = DataTools()
