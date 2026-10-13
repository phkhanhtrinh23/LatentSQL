"""A tiny self-contained Text-to-SQL fixture for the smoke test.

Two small SQLite databases and 30 question/SQL pairs, so the whole pipeline
(retrieval, teacher, curriculum, RL, evaluation) runs on any machine with no
download, no GPU and no API key. It checks plumbing only: never report
numbers from it.
"""
from __future__ import annotations

import os
import sqlite3
from typing import List, Tuple

from .loaders import Example

# --------------------------------------------------------------------------
# schema + rows
# --------------------------------------------------------------------------
_CONCERT_DDL = [
    """CREATE TABLE stadium (
        stadium_id INTEGER PRIMARY KEY,
        name TEXT,
        location TEXT,
        capacity INTEGER,
        average REAL
    )""",
    """CREATE TABLE singer (
        singer_id INTEGER PRIMARY KEY,
        name TEXT,
        country TEXT,
        age INTEGER,
        is_male INTEGER
    )""",
    """CREATE TABLE concert (
        concert_id INTEGER PRIMARY KEY,
        concert_name TEXT,
        theme TEXT,
        stadium_id INTEGER,
        year INTEGER,
        FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id)
    )""",
    """CREATE TABLE singer_in_concert (
        concert_id INTEGER,
        singer_id INTEGER,
        FOREIGN KEY (concert_id) REFERENCES concert(concert_id),
        FOREIGN KEY (singer_id) REFERENCES singer(singer_id)
    )""",
]

_CONCERT_ROWS = {
    "stadium": [
        (1, "Balmoor", "Peterhead", 4000, 1100.5),
        (2, "Recreation Park", "Alloa", 3100, 950.0),
        (3, "Glebe Park", "Brechin", 3960, 700.25),
        (4, "Gayfield Park", "Arbroath", 6600, 1400.0),
        (5, "Stark's Park", "Kirkcaldy", 10104, 2100.75),
    ],
    "singer": [
        (1, "Joe Sharp", "Netherlands", 52, 0),
        (2, "Timbaland", "United States", 32, 1),
        (3, "Justin Brown", "France", 29, 1),
        (4, "Rose White", "France", 41, 0),
        (5, "John Nizinik", "France", 43, 1),
        (6, "Tribal King", "France", 25, 1),
    ],
    "concert": [
        (1, "Auditions", "Free choice", 1, 2014),
        (2, "Super bootcamp", "Free choice 2", 2, 2014),
        (3, "Home Visits", "Bleeding Love", 2, 2015),
        (4, "Week 1", "Wide Awake", 3, 2014),
        (5, "Week 1", "Happy Tonight", 5, 2015),
        (6, "Week 2", "Party All Night", 5, 2015),
    ],
    "singer_in_concert": [
        (1, 2), (1, 3), (1, 5), (2, 3), (2, 6),
        (3, 5), (4, 4), (5, 6), (5, 3), (6, 2),
    ],
}

_SCHOOL_DDL = [
    """CREATE TABLE schools (
        school_id INTEGER PRIMARY KEY,
        school_name TEXT,
        county TEXT,
        charter INTEGER,
        enrollment INTEGER
    )""",
    """CREATE TABLE satscores (
        school_id INTEGER,
        num_takers INTEGER,
        avg_score_math INTEGER,
        avg_score_reading INTEGER,
        FOREIGN KEY (school_id) REFERENCES schools(school_id)
    )""",
]

_SCHOOL_ROWS = {
    "schools": [
        (1, "Alameda High", "Alameda", 0, 1600),
        (2, "Berkeley Charter", "Alameda", 1, 420),
        (3, "Fresno Central", "Fresno", 0, 2300),
        (4, "Lakeview Academy", "Fresno", 1, 380),
        (5, "Monterey Bay High", "Monterey", 0, 1100),
    ],
    "satscores": [
        (1, 300, 540, 520),
        (2, 90, 610, 600),
        (3, 410, 480, 470),
        (4, 75, 655, 640),
        (5, 210, 505, 512),
    ],
}

_DBS = {
    "concert_singer": (_CONCERT_DDL, _CONCERT_ROWS),
    "california_schools": (_SCHOOL_DDL, _SCHOOL_ROWS),
}


def build_databases(root: str) -> str:
    """Create ``<root>/database/<db_id>/<db_id>.sqlite`` and return the db root."""
    db_root = os.path.join(root, "database")
    for db_id, (ddl, rows) in _DBS.items():
        folder = os.path.join(db_root, db_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{db_id}.sqlite")
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            for stmt in ddl:
                cur.execute(stmt)
            for table, data in rows.items():
                if not data:
                    continue
                ph = ",".join(["?"] * len(data[0]))
                cur.executemany(f'INSERT INTO "{table}" VALUES ({ph})', data)
            conn.commit()
        finally:
            conn.close()
    return db_root


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------
# (db_id, question, gold_sql, evidence, difficulty)
_QUESTIONS: List[Tuple[str, str, str, str, str]] = [
    ("concert_singer", "How many stadiums are there?",
     "SELECT count(*) FROM stadium", "", "simple"),
    ("concert_singer", "What is the name of the stadium with the largest capacity?",
     "SELECT name FROM stadium ORDER BY capacity DESC LIMIT 1", "", "simple"),
    ("concert_singer", "List the names of all singers from France.",
     "SELECT name FROM singer WHERE country = 'France'", "", "simple"),
    ("concert_singer", "What is the average age of all singers?",
     "SELECT avg(age) FROM singer", "", "simple"),
    ("concert_singer", "How many singers are older than 40?",
     "SELECT count(*) FROM singer WHERE age > 40", "", "simple"),
    ("concert_singer", "Show the names and capacities of stadiums with capacity above 4000.",
     "SELECT name, capacity FROM stadium WHERE capacity > 4000", "", "simple"),
    ("concert_singer", "What are the distinct countries of the singers?",
     "SELECT DISTINCT country FROM singer", "", "simple"),
    ("concert_singer", "How many concerts happened in 2015?",
     "SELECT count(*) FROM concert WHERE year = 2015", "", "simple"),
    ("concert_singer", "Show the stadium name for each concert.",
     "SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 "
     "ON T1.stadium_id = T2.stadium_id", "", "moderate"),
    ("concert_singer", "Show the name of each stadium and the number of concerts held there.",
     "SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 "
     "ON T1.stadium_id = T2.stadium_id GROUP BY T2.name", "", "moderate"),
    ("concert_singer", "What is the name of the stadium that hosted the most concerts?",
     "SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 "
     "ON T1.stadium_id = T2.stadium_id GROUP BY T2.name ORDER BY count(*) DESC LIMIT 1",
     "", "moderate"),
    ("concert_singer", "List the names of singers who performed in at least two concerts.",
     "SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 "
     "ON T1.singer_id = T2.singer_id GROUP BY T2.name HAVING count(*) >= 2",
     "", "moderate"),
    ("concert_singer", "Show the names of singers who performed in the concert named 'Auditions'.",
     "SELECT T3.name FROM singer_in_concert AS T1 JOIN concert AS T2 "
     "ON T1.concert_id = T2.concert_id JOIN singer AS T3 ON T1.singer_id = T3.singer_id "
     "WHERE T2.concert_name = 'Auditions'", "", "moderate"),
    ("concert_singer", "What are the names of stadiums that have never hosted a concert?",
     "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
     "", "challenging"),
    ("concert_singer", "Show the names of singers who are older than the average singer age.",
     "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)", "", "challenging"),
    ("concert_singer", "List the countries that have more than two singers.",
     "SELECT country FROM singer GROUP BY country HAVING count(*) > 2", "", "moderate"),
    ("concert_singer", "What is the total capacity of stadiums located in Kirkcaldy?",
     "SELECT sum(capacity) FROM stadium WHERE location = 'Kirkcaldy'", "", "simple"),
    ("concert_singer", "Show the concert names ordered by year descending.",
     "SELECT concert_name FROM concert ORDER BY year DESC", "", "simple"),
    ("concert_singer", "How many singers did not perform in any concert?",
     "SELECT count(*) FROM singer WHERE singer_id NOT IN "
     "(SELECT singer_id FROM singer_in_concert)", "", "challenging"),
    ("concert_singer", "What is the maximum and minimum capacity among all stadiums?",
     "SELECT max(capacity), min(capacity) FROM stadium", "", "simple"),

    ("california_schools", "How many schools are in Alameda county?",
     "SELECT count(*) FROM schools WHERE county = 'Alameda'", "", "simple"),
    ("california_schools", "What is the name of the school with the highest math SAT score?",
     "SELECT T1.school_name FROM schools AS T1 JOIN satscores AS T2 "
     "ON T1.school_id = T2.school_id ORDER BY T2.avg_score_math DESC LIMIT 1",
     "highest math SAT score refers to max(avg_score_math)", "moderate"),
    ("california_schools", "List the names of charter schools.",
     "SELECT school_name FROM schools WHERE charter = 1",
     "charter school refers to charter = 1", "simple"),
    ("california_schools", "What is the average enrollment of schools in Fresno county?",
     "SELECT avg(enrollment) FROM schools WHERE county = 'Fresno'", "", "simple"),
    ("california_schools", "How many test takers are there in Alameda county?",
     "SELECT sum(T2.num_takers) FROM schools AS T1 JOIN satscores AS T2 "
     "ON T1.school_id = T2.school_id WHERE T1.county = 'Alameda'", "", "moderate"),
    ("california_schools", "Show the school names whose math score is above 600.",
     "SELECT T1.school_name FROM schools AS T1 JOIN satscores AS T2 "
     "ON T1.school_id = T2.school_id WHERE T2.avg_score_math > 600", "", "moderate"),
    ("california_schools", "Which county has the most schools?",
     "SELECT county FROM schools GROUP BY county ORDER BY count(*) DESC LIMIT 1",
     "", "moderate"),
    ("california_schools",
     "What is the average reading score of charter schools?",
     "SELECT avg(T2.avg_score_reading) FROM schools AS T1 JOIN satscores AS T2 "
     "ON T1.school_id = T2.school_id WHERE T1.charter = 1",
     "charter school refers to charter = 1", "moderate"),
    ("california_schools",
     "List schools with enrollment greater than the average enrollment.",
     "SELECT school_name FROM schools WHERE enrollment > (SELECT avg(enrollment) FROM schools)",
     "", "challenging"),
    ("california_schools", "How many counties are represented in the schools table?",
     "SELECT count(DISTINCT county) FROM schools", "", "simple"),
]


def load_synthetic(root: str, split: str = "train", limit: int | None = None) -> List[Example]:
    """Materialise the databases under ``root`` and return one split (train | test)."""
    from .loaders import random_split

    db_root = build_databases(root)
    out: List[Example] = []
    for i, (db_id, q, sql, ev, diff) in enumerate(_QUESTIONS):
        out.append(
            Example(
                qid=f"synthetic_{i}",
                db_id=db_id,
                question=q,
                gold_sql=sql,
                evidence=ev,
                dataset="synthetic",
                difficulty=diff,
                db_path=os.path.join(db_root, db_id, f"{db_id}.sqlite"),
            )
        )
    # a follow-up question with interaction history H, as in SParC / CoSQL
    out.append(Example(
        qid="synthetic_dialogue_1", db_id="concert_singer",
        question="Which of them are older than 40?",
        gold_sql="SELECT name FROM singer WHERE country = 'France' AND age > 40",
        history=["List the names of all singers from France."],
        dataset="synthetic", difficulty="moderate",
        db_path=os.path.join(db_root, "concert_singer", "concert_singer.sqlite"),
    ))
    train, test = random_split(out, heldout_fraction=0.3, seed=0)
    out = test if split == "test" else train
    return out[:limit] if limit else out
