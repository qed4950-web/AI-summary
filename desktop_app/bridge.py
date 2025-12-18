import subprocess
import os
import sys
import json

class Bridge:
    """
    Connects the GUI to the core logic via a persistent subprocess.
    This avoids reloading models for every query, significantly improving response time.
    """
    def __init__(self):
        self.process = None
        self._start_backend()

    def _start_backend(self):
        """
        Starts the RAG engine in interactive JSON mode.
        """
        # Get absolute path to scripts/run_doc.sh
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_doc_script = os.path.join(base_path, "scripts", "runners", "run_doc.sh")
        
        if not os.path.exists(run_doc_script):
            print(f"[BridgeError] Script not found: {run_doc_script}")
            return

        try:
            # Command: ./scripts/run_doc.sh --json (triggers interactive JSON mode)
            cmd = [run_doc_script, "--json"]
            
            # Spawn persistent process
            self.process = subprocess.Popen(
                cmd,
                cwd=base_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr, # Forward stderr to console for debugging
                text=True,
                bufsize=1  # Line buffered
            )
            print("[Bridge] Backend started (PID: {})".format(self.process.pid))
            
        except Exception as e:
            print(f"[BridgeError] Failed to start backend: {e}")

    def route(self, query: str, callback=None) -> str:
        """
        Routing logic:
        1. @photo -> Photo Agent
        2. @meeting -> Meeting Agent
        3. Simple Text -> RAG / Chat (via persistent backend)
        """
        # Strip prefix if it exists to clean up the query for RAG
        clean_query = query
        
        # Map UI @prefixes to Backend /commands
        if query.startswith("@사진") or query.startswith("@photo"):
            # Convert to /photo command
            clean_query = query.replace("@사진", "").replace("@photo", "").strip()
            return self.handle_chat(f"/photo {clean_query}", callback=callback)
            
        if query.startswith("@회의") or query.startswith("@meeting") or query.startswith("@녹음"):
            clean_query = query.replace("@회의", "").replace("@meeting", "").replace("@녹음", "").strip()
            return self.handle_chat(f"/meeting {clean_query}", callback=callback)

        if query.startswith("@문서") or query.startswith("@doc") or query.startswith("@검색"):
            # Convert to /search command
            clean_query = query.replace("@문서", "").replace("@doc", "").replace("@검색", "").strip()
            if not clean_query:
                return "검색할 내용을 입력해주세요. (예: @문서 계약서 요약)"
            return self.handle_chat(f"/search {clean_query}", callback=callback)
        
        # Default fallback is also Chat/RAG
        return self.handle_chat(query, callback=callback)

    # Removed handle_photo_agent as it is now routed to backend

    def handle_chat(self, query, callback=None):
        """
        Sends query to the persistent backend and reads the JSON response.
        Supports streaming updates if callback is provided.
        """
        if not self.process or self.process.poll() is not None:
            print("[Bridge] Process died, restarting...")
            self._start_backend()
            if not self.process:
                return "[System Error] Backend failed to start."

        try:
            # Write query to stdin
            if self.process.stdin:
                # Ensure query is single line so readline() reads it all at once
                sanitized_query = query.replace("\n", " ")
                self.process.stdin.write(sanitized_query + "\n")
                self.process.stdin.flush()
            
            # Read response from stdout (blocking loop)
            # Backend might print logs before the JSON, so we skip them.
            if self.process.stdout:
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        return "[Error] Backend closed connection."
                    
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Debug log to see what's coming from backend
                    # print(f"[Debug] Backend said: {line}")

                    # Try Parse JSON
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            
                            # Check for streaming event
                            if data.get("status") == "streaming" and callback:
                                content = data.get("content") or data.get("chunk")
                                if content:
                                    callback(content)
                                continue # Keep reading for final response
                            
                            answer = data.get("answer") or ""
                            
                            suggestions = data.get("suggestions") or []
                            if suggestions:
                                answer += "\n\n(Tip: " + ", ".join(suggestions) + ")"
                                
                            return answer
                        except json.JSONDecodeError:
                            # Might be a log line that coincidentally looks like JSON or broken JSON
                            pass
                    
                    # If not JSON, it's likely a log message (e.g. [LNPChat] ...)
                    # We just ignore it and wait for the next line.
                    print(f"[Backend Log] {line}")
                    
        except Exception as e:
            return f"[System Error] Communication failed: {str(e)}"
        
        return "[Error] Unknown state"

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
