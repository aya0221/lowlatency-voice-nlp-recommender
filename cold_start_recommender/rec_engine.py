import mysql.connector
from opensearchpy import OpenSearch
from datetime import datetime

class RecEngine:
    def __init__(self):
        """Initialize connections to MySQL and OpenSearch."""
        # Connect to MySQL database
        try:
            self.conn = mysql.connector.connect(
                host="localhost",
                user="aya",
                password="userdatatable__admin_aya",
                database="voice_ai_coach"
            )
            self.cursor = self.conn.cursor(dictionary=True)
        except mysql.connector.Error as err:
            raise Exception(f"Database connection failed: {err}")

        # Connect to OpenSearch
        try:
            self.search_client = OpenSearch(
                hosts=[{"host": "localhost", "port": 9200}],
                http_auth=("admin", "admin"),
                use_ssl=False,
                verify_certs=False
            )
        except Exception as err:
            print(f"Warning: OpenSearch connection failed: {err}")
            self.search_client = None

        # workout types mapping (for normalization) 
        self.known_types = {
            "cycling": "Cycling", "bike": "Cycling", "spin": "Cycling",
            "meditation": "Meditation", "meditate": "Meditation",
            "stretching": "Stretching", "stretch": "Stretching",
            "walking": "Walking", "walk": "Walking",
            "strength": "Strength", "strength training": "Strength",
            "yoga": "Yoga"
        }

        # Fitness level normalization
        self.levels = {
            "beginner": "Beginner",
            "intermediate": "Intermediate",
            "advanced": "Advanced"
        }

        # Common stopwords to ignore in query parsing
        self.stopwords = {
            "find", "me", "a", "an", "the", "for", "give", "show", "looking",
            "workout", "workouts", "session", "sessions", "exercise", "exercises",
            "want", "recommend", "i", "training"
        }

    def _parse_query(self, query_text):
        """Parse the query text and extract keywords."""
        text = query_text.lower()
        tokens = text.split()
        filter_type = None
        filter_level = None
        duration_range = None
        keywords = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in self.known_types:
                filter_type = self.known_types[token]
                i += 1
                continue
            if token in self.levels:
                filter_level = self.levels[token]
                i += 1
                continue
            if token.isdigit():
                num = int(token)
                if i+1 < len(tokens) and tokens[i+1] in ["minute", "minutes", "min", "mins", "hour", "hours"]:
                    unit = tokens[i+1]
                    i += 2
                    if unit.startswith("hour"):
                        num *= 60
                else:
                    i += 1
                low = max(int(num * 0.8), 1)
                high = int(num * 1.2)
                duration_range = (low, high)
                continue
            if token in ["minute", "minutes", "min", "mins", "hour", "hours"]:
                i += 1
                continue
            if token in self.stopwords:
                i += 1
                continue
            keywords.append(token)
            i += 1
        keyword_query = " ".join(keywords) if keywords else None
        return filter_type, filter_level, duration_range, keyword_query

    def get_recommendations(self, user_id, query_text=None):
        """Generate personalized workout recommendations for the user."""
        # Fetch user profile
        self.cursor.execute("SELECT age_group, fitness_level, workout_types FROM users WHERE user_id = %s", (user_id,))
        user = self.cursor.fetchone()
        if not user:
            raise Exception(f"User {user_id} not found. Please onboard the user first.")
        age_group = user["age_group"]
        fitness_level = user["fitness_level"]
        # List of preferred workout types from profile
        types_pref = [t.strip() for t in (user["workout_types"] or "").split(",") if t.strip()]
        # Check if user has any past sessions
        self.cursor.execute("SELECT COUNT(*) as cnt FROM sessions WHERE user_id = %s", (user_id,))
        session_count = self.cursor.fetchone()["cnt"]
        has_history = session_count and session_count > 0

        # Parse query text into filters
        if not query_text or query_text.strip() == "":
            query_text = ""  # treat None/empty as an empty query (for parsing)
        filter_type, filter_level, duration_range, keyword_query = self._parse_query(query_text)
        # If query has no meaningful content (just generic request)
        if not filter_type and not filter_level and not duration_range and not keyword_query:
            # No specific query, provide default recommendations
            if not has_history:
                # First-time user with no history — recommend top-ranked workouts for their age group, fitness level, and preferred workout types
                results = [] 
                segment_types = types_pref if types_pref else [None] # Use preferred workout types if specified, else run once with no type filter
                for pref_type in segment_types:
                    if pref_type:
                        # Look up the top 3 recommended workouts for users who match this age group, fitness level, and workout type preference
                        self.cursor.execute(
                            "SELECT workout_id FROM cold_start_top_workouts "
                            "WHERE age_group = %s AND fitness_level = %s AND workout_type = %s "
                            "ORDER BY rank LIMIT 3",
                            (age_group, fitness_level, pref_type)
                        )
                    else:
                        # No preferred type: get top 3 general workouts for (age_group, fitness_level) only
                        self.cursor.execute(
                            "SELECT workout_id FROM cold_start_top_workouts "
                            "WHERE age_group = %s AND fitness_level = %s ORDER BY rank LIMIT 3",
                            (age_group, fitness_level)
                        )
                    # Go through each workout returned from the query to filter and finalize the recommendation list
                    for row in self.cursor.fetchall():
                        wid = row["workout_id"]
                        # Look up the most recent instructor who led this workout across all users (via sessions table)
                        # → used to check if this instructor is globally flagged and should be excluded from results
                        self.cursor.execute(
                            "SELECT instructor_id FROM sessions WHERE workout_id = %s ORDER BY started_at DESC LIMIT 1",
                            (wid,)
                        )
                        instr_row = self.cursor.fetchone() 
                        instr = instr_row["instructor_id"] if instr_row else None
                        # if instr: # remove global filtering  
                        #     self.cursor.execute(
                        #         "SELECT * FROM flags WHERE component_type='instructor' AND component_id = %s AND flagged=1 AND manual_override=0",
                        #         (instr,)
                        #     )
                        #     if self.cursor.fetchone():
                        #         continue
                        results.append(wid)
                # Remove duplicates and return
                return list(dict.fromkeys(results))  # preserve order, remove duplicates
            else:
                # Personalized defaults for returning user with no new query — use past feedback or recent activity
                recs = []
                # Step 1: Try to infer user’s workout preference based on their feedback history
                # → Find the most recent workout type that the user gave a thumbs-up to
                self.cursor.execute(
                    "SELECT s.workout_type FROM sessions s JOIN feedback f ON s.session_id=f.session_id "
                    "WHERE f.feedback=1 AND s.user_id=%s ORDER BY f.feedback_time DESC LIMIT 1",
                    (user_id,)
                )
                row = self.cursor.fetchone()
                suggest_type = row["workout_type"] if row else None
                # Step 2: If no thumbs-up feedback found, fall back to user’s profile or recent completion
                if not suggest_type:
                    if types_pref:
                        suggest_type = types_pref[0]
                    else:
                        self.cursor.execute(
                            "SELECT workout_type FROM sessions WHERE user_id=%s AND completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1",
                            (user_id,)
                        )
                        row2 = self.cursor.fetchone()
                        if row2:
                            suggest_type = row2["workout_type"]
                # Step 3: Fetch top recommended workouts for user’s segment, filtered by the suggested type if available
                if suggest_type:
                    self.cursor.execute(
                        "SELECT workout_id FROM cold_start_top_workouts "
                        "WHERE age_group=%s AND fitness_level=%s AND workout_type=%s ORDER BY rank LIMIT 3",
                        (age_group, fitness_level, suggest_type)
                    )
                    recs = [r["workout_id"] for r in self.cursor.fetchall()]
                else:
                    # If no suggestion type can be inferred, just fetch top general workouts for user segment
                    self.cursor.execute(
                        "SELECT workout_id FROM cold_start_top_workouts "
                        "WHERE age_group=%s AND fitness_level=%s ORDER BY rank LIMIT 3",
                        (age_group, fitness_level)
                    )
                    recs = [r["workout_id"] for r in self.cursor.fetchall()]
                # Filter out flagged instructors
                filtered_recs = []
                for wid in recs:
                    self.cursor.execute(
                        "SELECT instructor_id FROM sessions WHERE workout_id = %s ORDER BY started_at DESC LIMIT 1",
                        (wid,)
                    )
                    instr_row = self.cursor.fetchone()
                    instr = instr_row["instructor_id"] if instr_row else None
                    if instr:
                        self.cursor.execute(
                            "SELECT * FROM flags WHERE component_type='instructor' AND component_id = %s AND flagged=1 AND manual_override=0",
                            (instr,)
                        )
                        if self.cursor.fetchone():
                            continue
                    filtered_recs.append(wid)
                return filtered_recs

        # Otherwise, we have some specific query to process via OpenSearch
        if not self.search_client:
            # If OpenSearch is not available, fall back to generic recommendations
            return self.get_recommendations(user_id, "")

        # Build OpenSearch query with personalization
        query_filters = []
        must_clauses = []
        should_clauses = []
        must_not_clauses = []

        # Apply filters from query or profile
        if filter_type:
            query_filters.append({"term": {"workout_type": filter_type}})
        else:
            # No specific type in query: boost user's preferred types
            for t in types_pref:
                if t:
                    should_clauses.append({"term": {"workout_type": {"value": t, "boost": 2.0}}})
        if filter_level:
            query_filters.append({"term": {"fitness_level": filter_level}})
        else:
            # No level specified: default to user's fitness level as filter
            if fitness_level:
                query_filters.append({"term": {"fitness_level": fitness_level}})
        if duration_range:
            low, high = duration_range
            query_filters.append({"range": {"duration": {"gte": low, "lte": high}}})

        # Prepare must_not clauses for flagged or disliked content
        # Global flagged instructors:
        self.cursor.execute("SELECT component_id FROM flags WHERE component_type='instructor' AND flagged=1 AND manual_override=0")
        flagged_instrs = [row["component_id"] for row in self.cursor.fetchall()]
        # User's liked/disliked instructors:
        self.cursor.execute(
            "SELECT DISTINCT s.instructor_id FROM sessions s JOIN feedback f ON s.session_id=f.session_id "
            "WHERE f.feedback=1 AND s.user_id=%s", (user_id,)
        )
        liked_instrs = [row["instructor_id"] for row in self.cursor.fetchall()]
        self.cursor.execute(
            "SELECT DISTINCT s.instructor_id FROM sessions s JOIN feedback f ON s.session_id=f.session_id "
            "WHERE f.feedback=-1 AND s.user_id=%s", (user_id,)
        )
        disliked_instrs = [row["instructor_id"] for row in self.cursor.fetchall()]
        # Workouts the user has already seen (to avoid repeats):
        self.cursor.execute("SELECT DISTINCT workout_id FROM sessions WHERE user_id=%s", (user_id,))
        seen_workouts = [str(row["workout_id"]) for row in self.cursor.fetchall()]

        for instr in set(flagged_instrs + disliked_instrs):
            must_not_clauses.append({"term": {"instructor_id": instr}})
        if seen_workouts:
            must_not_clauses.append({"terms": {"_id": seen_workouts}})
        # Boost liked instructors in results
        for instr in liked_instrs:
            should_clauses.append({"term": {"instructor_id": {"value": instr, "boost": 3.0}}})

        # Text search clause for remaining keywords
        if keyword_query:
            must_clauses.append({
                "multi_match": {
                    "query": keyword_query,
                    "fields": ["title^2", "description", "tags"]
                }
            })
        if not must_clauses:
            must_clauses.append({"match_all": {}})  # no specific keywords, match all (within filters)

        # Assemble the OpenSearch query
        query_body = {
            "size": 5,
            "query": {
                "bool": {
                    "filter": query_filters,
                    "must": must_clauses,
                    "should": should_clauses,
                    "must_not": must_not_clauses
                }
            }
        }
        # Execute the search query:contentReference[oaicite:5]{index=5}
        try:
            response = self.search_client.search(index="workouts", body=query_body)
        except Exception as e:
            print(f"WARNING: OpenSearch query failed, using cold-start fallback. Error: {e}")
            return self.get_recommendations(user_id, "")
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            # Get workout ID from result
            wid = hit.get("_id") or hit.get("_source", {}).get("id") or hit.get("_source", {}).get("workout_id")
            if not wid:
                continue
            instr = hit.get("_source", {}).get("instructor_id")
            # Double-check that the instructor isn’t flagged (should already be filtered out)
            if instr in flagged_instrs:
                continue
            results.append(wid)
        return results

    def start_session(self, user_id, workout_id, workout_type, instructor_id):
        """Log the start of a workout session."""
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO sessions (user_id, workout_id, workout_type, instructor_id, started_at, completed_at) "
            "VALUES (%s, %s, %s, %s, %s, NULL)",
            (user_id, workout_id, workout_type, instructor_id, started)
        )
        self.conn.commit()
        session_id = self.cursor.lastrowid
        return session_id

    def end_session(self, session_id, completed=True):
        """Mark a session as completed or abandoned."""
        if completed:
            finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute("UPDATE sessions SET completed_at=%s WHERE session_id=%s", (finished, session_id))
            self.conn.commit()
        else:
            # Session abandoned
            self.check_and_flag(session_id)
        # (If needed, could add logic to record duration or other metrics)

    def log_feedback(self, session_id, feedback):
        """Log thumbs up/down feedback for a session. (feedback=1 for up, -1 for down)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Retrieve user_id and workout_id for reference
        self.cursor.execute("SELECT user_id, workout_id FROM sessions WHERE session_id=%s", (session_id,))
        row = self.cursor.fetchone()
        if not row:
            raise Exception("Session not found for feedback.")
        user_id = row["user_id"]; workout_id = row["workout_id"]
        self.cursor.execute(
            "INSERT INTO feedback (session_id, user_id, workout_id, feedback, feedback_time) VALUES (%s, %s, %s, %s, %s)",
            (session_id, user_id, workout_id, feedback, timestamp)
        )
        self.conn.commit()

    def check_and_flag(self, session_id):
        """Auto-flag instructors with >30% abandonment rate across >=3 sessions."""
        # Identify instructor for this session
        self.cursor.execute("SELECT instructor_id FROM sessions WHERE session_id=%s", (session_id,))
        row = self.cursor.fetchone()
        if not row:
            return
        instructor_id = row["instructor_id"]
        # Calculate abandonment stats for this instructor
        self.cursor.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END) as abandoned "
            "FROM sessions WHERE instructor_id=%s",
            (instructor_id,)
        )
        stats = self.cursor.fetchone()
        total = stats["total"]; abandoned = stats["abandoned"] or 0
        if total >= 3:
            rate = abandoned / total
            if rate > 0.30:
                # Flag the instructor if not already flagged
                self.cursor.execute(
                    "SELECT * FROM flags WHERE component_type='instructor' AND component_id=%s",
                    (instructor_id,)
                )
                existing = self.cursor.fetchone()
                if existing:
                    if existing.get("manual_override"):
                        return  # skip auto-flag if manual override is set
                    if existing.get("flagged") == 0:
                        # Update existing record to flagged
                        self.cursor.execute(
                            "UPDATE flags SET flagged=1, flag_reason=%s, flagged_at=%s WHERE flag_id=%s",
                            (f"abandonment_rate>{rate:.0%}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), existing["flag_id"])
                        )
                        self.conn.commit()
                else:
                    # Insert new flag record
                    self.cursor.execute(
                        "INSERT INTO flags (component_type, component_id, flagged, manual_override, flag_reason, flagged_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        ("instructor", instructor_id, 1, 0, f"abandonment_rate>{rate:.0%}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    self.conn.commit()
