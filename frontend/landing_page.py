import os
import json
import signal
import sys
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from pathlib import Path
from datetime import datetime, timedelta
from retrieval import POCRetriever

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "similarity_check"))

from langchain.schema import Document
from ingestion import chunk_documents, ingest_documents
from embedding import download_hugging_face_embeddings

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this to a strong secret key


# Graceful shutdown handler
def shutdown_handler(signum, frame):
	print("\n\n🛑 Shutting down gracefully...")
	sys.exit(0)


# Register signal handlers for Ctrl+C
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Configure paths
EXCEPTION_DIR = os.path.join(os.path.dirname(__file__), "exception_poc")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "poc_files")
MANAGERS_FILE = os.path.join(os.path.dirname(__file__), "managers.json")
EMPLOYEES_FILE = os.path.join(os.path.dirname(__file__), "employees.json")

# Initialize POCRetriever
try:
	retriever = POCRetriever(index_name="innoscan", knowledge_base_path=KNOWLEDGE_BASE_PATH)
	print("✅ POCRetriever initialized successfully")
except Exception as e:
	print(f"⚠️ Warning: Could not initialize POCRetriever: {e}")
	retriever = None

# Initialize embeddings for ingestion
embeddings_model = None
try:
	embeddings_model = download_hugging_face_embeddings()
	print("✅ Embeddings model loaded for ingestion")
except Exception as e:
	print(f"⚠️ Warning: Could not load embeddings model: {e}")
	embeddings_model = None


def load_managers():
	"""Load manager credentials from JSON file."""
	if os.path.exists(MANAGERS_FILE):
		try:
			with open(MANAGERS_FILE, 'r', encoding='utf-8') as f:
				return json.load(f)
		except Exception as e:
			print(f"⚠️ Error loading managers: {e}")
	return {}


def init_default_managers():
	"""Initialize default manager credentials if file doesn't exist."""
	if not os.path.exists(MANAGERS_FILE):
		default_managers = {
			"managers": [
				{"id": "manager1", "password": "password123", "name": "Manager One"},
				{"id": "manager2", "password": "password123", "name": "Manager Two"},
				{"id": "admin", "password": "admin123", "name": "Admin"}
			]
		}
		try:
			with open(MANAGERS_FILE, 'w', encoding='utf-8') as f:
				json.dump(default_managers, f, indent=2)
			print("✅ Default managers file created")
		except Exception as e:
			print(f"⚠️ Error creating managers file: {e}")


def load_employees():
	"""Load employee credentials from JSON file."""
	if os.path.exists(EMPLOYEES_FILE):
		try:
			with open(EMPLOYEES_FILE, 'r', encoding='utf-8') as f:
				return json.load(f)
		except Exception as e:
			print(f"⚠️ Error loading employees: {e}")
	return {}


def init_default_employees():
	"""Initialize default employee credentials if file doesn't exist."""
	if not os.path.exists(EMPLOYEES_FILE):
		default_employees = {
			"employees": [
				{"id": "emp001", "password": "password123", "name": "Employee One"},
				{"id": "emp002", "password": "password123", "name": "Employee Two"},
				{"id": "emp003", "password": "password123", "name": "Employee Three"}
			]
		}
		try:
			with open(EMPLOYEES_FILE, 'w', encoding='utf-8') as f:
				json.dump(default_employees, f, indent=2)
			print("✅ Default employees file created")
		except Exception as e:
			print(f"⚠️ Error creating employees file: {e}")


# Initialize default managers and employees on startup
init_default_managers()
init_default_employees()


def get_idea_title(idea_id):
	"""Get the title of an idea from the uploads directory."""
	try:
		idea_file = os.path.join(UPLOADS_DIR, f"{idea_id}.json")
		if os.path.exists(idea_file):
			with open(idea_file, 'r', encoding='utf-8') as f:
				idea_data = json.load(f)
			return idea_data.get("title", "N/A")
	except Exception as e:
		print(f"⚠️ Error getting idea title for {idea_id}: {e}")
	return "N/A"


def save_poc_to_knowledge_base(poc_record):
	"""
	Save a POC record to the local knowledge base.
	
	Args:
		poc_record: Dictionary containing POC data with 'id' field
	
	Returns:
		tuple: (success: bool, file_path: str, message: str)
	"""
	try:
		# Ensure knowledge base directory exists
		os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
		
		poc_id = poc_record.get("id")
		if not poc_id:
			return False, None, "❌ POC record missing 'id' field"
		
		# Create file path
		kb_file_path = os.path.join(KNOWLEDGE_BASE_PATH, f"{poc_id}.json")
		
		# Save to knowledge base
		with open(kb_file_path, "w", encoding="utf-8") as f:
			json.dump(poc_record, f, indent=2, ensure_ascii=False)
		
		return True, kb_file_path, f"✅ POC saved to knowledge base: {kb_file_path}"
	
	except Exception as e:
		error_msg = f"❌ Error saving POC to knowledge base: {str(e)}"
		print(error_msg)
		return False, None, error_msg


@app.route('/')
def index():
	"""Render the login page"""
	return render_template('login.html')


@app.route('/manager')
def manager():
	"""Render the manager dashboard (requires login)"""
	if 'manager_id' not in session:
		return redirect(url_for('index'))
	return render_template('manager_dashboard.html')


@app.route('/check-exception-status', strict_slashes=False)
def check_exception_status():
	return render_template('check_exception_status.html')


@app.route('/api/get-exception-status', methods=['GET'])
def get_exception_status():
	"""
	Retrieve exception request status by exception ID.
	"""
	exception_id = request.args.get('exception_id', '').strip()
	
	if not exception_id:
		return jsonify({
			"success": False,
			"message": "Exception ID is required"
		}), 400
	
	try:
		exception_file = os.path.join(EXCEPTION_DIR, f"{exception_id}.json")
		
		if not os.path.exists(exception_file):
			return jsonify({
				"success": False,
				"message": f"❌ Exception ID '{exception_id}' not found. Please check the ID and try again."
			}), 404
		
		# Read exception data
		with open(exception_file, "r", encoding="utf-8") as f:
			exception_data = json.load(f)
		
		return jsonify({
			"success": True,
			"message": "✅ Exception request found",
			"data": exception_data
		}), 200
	
	except json.JSONDecodeError:
		return jsonify({
			"success": False,
			"message": "Error reading exception data. File may be corrupted."
		}), 500
	
	except Exception as e:
		return jsonify({
			"success": False,
			"message": f"Error retrieving exception status: {str(e)}"
		}), 500


@app.route('/api/manager-login', methods=['POST'])
def manager_login():
	"""Authenticate manager and create session."""
	try:
		data = request.get_json()
		manager_id = data.get('manager_id', '').strip()
		password = data.get('password', '').strip()

		if not manager_id or not password:
			return jsonify({
				"success": False,
				"message": "Manager ID and password are required"
			}), 400

		# Load managers
		managers_data = load_managers()
		managers = managers_data.get('managers', [])

		# Find matching manager
		for manager in managers:
			if manager.get('id') == manager_id and manager.get('password') == password:
				# Create session
				session.permanent = True
				session['manager_id'] = manager_id
				session['manager_name'] = manager.get('name', 'Manager')
				
				print(f"✅ Manager '{manager_id}' logged in")
				return jsonify({
					"success": True,
					"message": "Login successful"
				}), 200

		# Invalid credentials
		print(f"❌ Failed login attempt for manager_id: {manager_id}")
		return jsonify({
			"success": False,
			"message": "Invalid manager ID or password"
		}), 401

	except Exception as e:
		print(f"⚠️ Error in manager login: {e}")
		return jsonify({
			"success": False,
			"message": f"An error occurred: {str(e)}"
		}), 500


@app.route('/api/logout', methods=['POST'])
def logout():
	"""Logout manager and clear session."""
	manager_id = session.get('manager_id')
	session.clear()
	print(f"✅ Manager '{manager_id}' logged out")
	return jsonify({
		"success": True,
		"message": "Logged out successfully"
	}), 200


@app.route('/api/manager-profile', methods=['GET'])
def manager_profile():
	"""Get current logged-in manager's profile."""
	try:
		if 'manager_id' not in session:
			return jsonify({
				"success": False,
				"message": "Not logged in"
			}), 401

		manager_id = session.get('manager_id')
		manager_name = session.get('manager_name', 'Manager')

		return jsonify({
			"success": True,
			"data": {
				"manager_id": manager_id,
				"manager_name": manager_name
			}
		}), 200

	except Exception as e:
		print(f"⚠️ Error getting manager profile: {e}")
		return jsonify({
			"success": False,
			"message": f"Error: {str(e)}"
		}), 500


@app.route('/employee')
def employee():
	"""Render the employee form (requires login)"""
	if 'employee_id' not in session:
		return redirect(url_for('index'))
	return render_template('form.html')


@app.route('/api/managers', methods=['GET'])
def get_managers():
	"""Get list of all managers for dropdown selection."""
	try:
		managers_data = load_managers()
		managers = managers_data.get('managers', [])
		
		# Return only id and name (not password)
		manager_list = [
			{"id": m.get('id'), "name": m.get('name', 'Unknown')} 
			for m in managers
		]
		
		return jsonify({
			"success": True,
			"data": manager_list
		}), 200
	except Exception as e:
		print(f"⚠️ Error getting managers: {e}")
		return jsonify({
			"success": False,
			"message": f"Error: {str(e)}"
		}), 500


@app.route('/api/submit', methods=['POST'])
def submit_idea():
	"""Submit an idea and check for similar POCs."""
	import uuid
	data = request.get_json()
	
	# Validate all required fields
	empty_fields = []
	for field in ["title", "description", "problem", "outcome", "language", "approach", "stack", "complexity", "timeline", "manager"]:
		if not data.get(field) or not str(data.get(field)).strip():
			field_name = field.replace("_", " ").title()
			empty_fields.append(field_name)
	
	if not data.get("skills") or len(data.get("skills", [])) == 0:
		empty_fields.append("Required Skills / Roles")
	
	if empty_fields:
		return jsonify({
			"success": False, 
			"message": f"Please fill in the following required fields: {', '.join(empty_fields)}",
			"errors": empty_fields
		}), 400
	
	# Generate ID
	idea_id = str(uuid.uuid4())[:8]
	
	# Create summary data
	summary_data = {
		"id": idea_id,
		"title": data.get("title"),
		"description": data.get("description"),
		"problem": data.get("problem"),
		"outcome": data.get("outcome"),
		"language": data.get("language"),
		"approach": data.get("approach"),
		"stack": data.get("stack"),
		"complexity": data.get("complexity"),
		"boilerplate_enabled": data.get("boilerplate_enabled", False),
		"dev_count": int(data.get("dev_count", 1)),
		"skills": data.get("skills", []),
		"timeline": data.get("timeline"),
		"manager": data.get("manager"),
	}
	
	# Save to uploads directory
	record_file = os.path.join(UPLOADS_DIR, f"{idea_id}.json")
	with open(record_file, "w", encoding="utf-8") as f:
		json.dump(summary_data, f, indent=2, ensure_ascii=False)
	print(f"✅ Saved to uploads: {record_file}")
	
	# Perform similarity check (only save to KB if no similar POC found)
	similar_pocs = []
	matched_poc_details = None
	similarity_score = None
	ingestion_status = None
	
	print(f"\n📋 Submitted Idea ID: {idea_id}")
	
	if retriever:
		try:
			similar_pocs = retriever.find_similar_pocs(
				title=data.get("title", ""),
				description=data.get("description", ""),
				problem=data.get("problem", ""),
				score_threshold=0.7,
				top_k=1
			)
			print(f"DEBUG: Similar POCs found: {len(similar_pocs)}")
			
			if similar_pocs:
				# ✅ SIMILAR POC FOUND - Return details but DON'T create exception yet
				top_poc = similar_pocs[0]
				poc_id = top_poc.get('poc_id')
				similarity_score = top_poc.get('score')
				print(f"✅ Most Similar POC: {poc_id} (Score: {similarity_score})")
				
				matched_poc_details = retriever.get_poc_from_knowledge_base(poc_id)
				if matched_poc_details:
					print(f"📄 POC Details: {matched_poc_details.get('title')}")
					print(f"⏳ Waiting for user confirmation via notes form...")
					
					# Build informative message with similar POC details
					similar_title = matched_poc_details.get('title', 'Unknown')
					similarity_percentage = similarity_score * 100
					message = f"⚠️ Similar POC found: '{similar_title}' (ID: {poc_id}) with {similarity_percentage:.1f}% similarity. Please review and decide if you want to proceed."
					
					# Return response with similar POC info
					# Frontend will show popup form asking for notes
					return jsonify({
						"success": True,
						"id": idea_id,
						"message": message,
						"data": summary_data,
						"file": record_file,
						"similar_poc_found": True,
						"similar_poc": matched_poc_details,
						"similarity_score": similarity_score,
						"similarity_percentage": similarity_percentage,
						"similar_poc_id": poc_id,
						"notes_required": True
					}), 200
				else:
					print(f"❌ Failed to load POC details from knowledge base")
					# Still found similar POC - return message and DON'T save to KB
					return jsonify({
						"success": True,
						"id": idea_id,
						"message": "⚠️ Similar POC found but details unavailable. Please try again.",
						"data": summary_data,
						"file": record_file,
						"similar_poc_found": True,
						"similar_poc_id": poc_id,
						"notes_required": False
					}), 200
			else:
				# ✅ NO SIMILAR POC - Auto-ingest to Pinecone AND save to knowledge base
				print(f"⚠️ No similar POCs found above threshold")
				
				# Save to knowledge base only if no similar POC found
				kb_success, kb_path, kb_msg = save_poc_to_knowledge_base(summary_data)
				print(kb_msg)
				
				if embeddings_model:
					try:
						print(f"\n🔄 Ingesting new POC to Pinecone...")
						
						# Prepare the submission data as a POC record
						new_poc_record = {
							"id": idea_id,
							"title": data.get("title"),
							"description": data.get("description"),
							"problem": data.get("problem"),
							"outcome": data.get("outcome"),
							"language": data.get("language"),
							"approach": data.get("approach"),
							"stack": data.get("stack"),
							"complexity": data.get("complexity"),
							"skills": data.get("skills", []),
							"timeline": data.get("timeline"),
							"manager": data.get("manager"),
							"boilerplate_enabled": data.get("boilerplate_enabled", False),
							"dev_count": data.get("dev_count", 1)
						}
						
						# Chunk and ingest
						chunks = chunk_documents([new_poc_record], chunk_size=500, chunk_overlap=20)
						if chunks:
							docsearch = ingest_documents(chunks, index_name="innoscan", embeddings=embeddings_model)
							ingestion_status = f"✅ New POC ingested successfully to Pinecone (ID: {idea_id})"
							print(ingestion_status)
						else:
							ingestion_status = "⚠️ Failed to create chunks for ingestion"
							print(ingestion_status)
					except Exception as e:
						ingestion_status = f"⚠️ Ingestion failed: {str(e)}"
						print(ingestion_status)
						import traceback
						traceback.print_exc()
				
				# Return success response for new POC
				response_message = f"✅ New POC submitted successfully!"
				if ingestion_status and "successfully" in ingestion_status.lower():
					response_message += f" {ingestion_status}"
				
				return jsonify({
					"success": True,
					"id": idea_id,
					"message": response_message,
					"data": summary_data,
					"file": record_file,
					"similar_poc_found": False,
					"ingestion_status": ingestion_status
				}), 201
		
		except Exception as e:
			print(f"⚠️ Retriever error: {e}")
			import traceback
			traceback.print_exc()
			
			# If retriever fails, treat as new POC
			return jsonify({
				"success": True,
				"id": idea_id,
				"message": "✅ Idea submitted (similarity check unavailable)",
				"data": summary_data,
				"file": record_file,
				"similar_poc_found": False
			}), 201
	
	# If retriever not initialized, treat as new POC
	return jsonify({
		"success": True,
		"id": idea_id,
		"message": "✅ Idea submitted successfully!",
		"data": summary_data,
		"file": record_file,
		"similar_poc_found": False
	}), 201


@app.route('/api/request-exception', methods=['POST'])
def request_exception():
	"""Handle exception request when user wants to proceed with similar idea."""
	import uuid
	data = request.get_json()
	
	# Validate required fields
	required_fields = ["idea_id", "manager", "similar_poc_id", "similarity_score"]
	missing_fields = [f for f in required_fields if not data.get(f)]
	
	# Check if notes are provided
	notes = data.get("notes", "").strip()
	if not notes:
		missing_fields.append("Notes")
	
	if missing_fields:
		return jsonify({
			"success": False,
			"message": f"Missing required fields: {', '.join(missing_fields)}"
		}), 400
	
	# Get title from idea_id
	idea_id = data.get("idea_id")
	idea_title = "N/A"
	try:
		idea_file = os.path.join(UPLOADS_DIR, f"{idea_id}.json")
		if os.path.exists(idea_file):
			with open(idea_file, "r", encoding="utf-8") as f:
				idea_data = json.load(f)
			idea_title = idea_data.get("title", "N/A")
	except Exception as e:
		print(f"⚠️ Error getting idea title: {e}")
	
	# Generate exception ID
	exception_id = str(uuid.uuid4())[:8]
	
	# Create exception request
	exception_data = {
		"id": exception_id,
		"idea_id": idea_id,
		"title": idea_title,
		"manager": data.get("manager"),
		"similar_poc_id": data.get("similar_poc_id"),
		"similarity_score": float(data.get("similarity_score")),
		"similarity_percentage": float(data.get("similarity_score")) * 100,
		"status": "pending",
		"created_at": datetime.now().isoformat(),
		"notes": notes
	}
	
	# Save exception request
	exception_file = os.path.join(EXCEPTION_DIR, f"{exception_id}.json")
	with open(exception_file, "w", encoding="utf-8") as f:
		json.dump(exception_data, f, indent=2, ensure_ascii=False)
	
	print(f"\n📋 Exception Request Created:")
	print(f"  Exception ID: {exception_id}")
	print(f"  Manager: {data.get('manager')}")
	
	return jsonify({
		"success": True,
		"exception_id": exception_id,
		"message": f"✅ Exception request submitted for manager review (ID: {exception_id})",
		"data": exception_data,
		"file": exception_file
	}), 201


@app.route('/api/employee-login', methods=['POST'])
def employee_login():
	"""Authenticate employee and create session."""
	try:
		data = request.get_json()
		employee_id = data.get('employee_id', '').strip()
		password = data.get('password', '').strip()

		if not employee_id or not password:
			return jsonify({
				"success": False,
				"message": "Employee ID and password are required"
			}), 400

		# Load employees
		employees_data = load_employees()
		employees = employees_data.get('employees', [])

		# Find matching employee
		for employee in employees:
			if employee.get('id') == employee_id and employee.get('password') == password:
				# Create session
				session.permanent = True
				session['employee_id'] = employee_id
				session['employee_name'] = employee.get('name', 'Employee')
				
				print(f"✅ Employee '{employee_id}' logged in")
				return jsonify({
					"success": True,
					"message": "Login successful"
				}), 200

		# Invalid credentials
		print(f"❌ Failed login attempt for employee_id: {employee_id}")
		return jsonify({
			"success": False,
			"message": "Invalid employee ID or password"
		}), 401

	except Exception as e:
		print(f"⚠️ Error in employee login: {e}")
		return jsonify({
			"success": False,
			"message": f"An error occurred: {str(e)}"
		}), 500


@app.route('/api/employee-profile', methods=['GET'])
def employee_profile():
	"""Get current logged-in employee's profile."""
	try:
		if 'employee_id' not in session:
			return jsonify({
				"success": False,
				"message": "Not logged in"
			}), 401

		employee_id = session.get('employee_id')
		employee_name = session.get('employee_name', 'Employee')

		return jsonify({
			"success": True,
			"data": {
				"employee_id": employee_id,
				"employee_name": employee_name
			}
		}), 200

	except Exception as e:
		print(f"⚠️ Error getting employee profile: {e}")
		return jsonify({
			"success": False,
			"message": f"Error: {str(e)}"
		}), 500


@app.route('/api/exceptions', methods=['GET'])
def get_exceptions():
	"""
	Get list of all exceptions with basic info (ID, title, status, manager).
	"""
	try:
		exceptions = []
		
		if not os.path.exists(EXCEPTION_DIR):
			return jsonify({
				"success": True,
				"data": []
			}), 200
		
		# Load all exception JSON files
		for filename in os.listdir(EXCEPTION_DIR):
			if filename.endswith('.json'):
				filepath = os.path.join(EXCEPTION_DIR, filename)
				try:
					with open(filepath, 'r', encoding='utf-8') as f:
						exception_data = json.load(f)
					
					# Get title from exception JSON, fallback to idea title if not present
					title = exception_data.get("title", "N/A")
					if title == "N/A":
						idea_id = exception_data.get("idea_id", "")
						title = get_idea_title(idea_id) if idea_id else "N/A"
					
					# Extract basic info for list view
					exception_info = {
						"id": exception_data.get("id", filename.replace('.json', '')),
						"idea_id": exception_data.get("idea_id", ""),
						"title": title,
						"status": exception_data.get("status", "pending"),
						"manager": exception_data.get("manager", "Unassigned"),
						"similarity_score": exception_data.get("similarity_score", 0),
						"similar_poc_id": exception_data.get("similar_poc_id", "N/A"),
						"created_at": exception_data.get("created_at", "N/A")
					}
					exceptions.append(exception_info)
				except Exception as e:
					print(f"⚠️ Error reading {filename}: {e}")
					continue
		
		# Sort by creation date (newest first)
		exceptions.sort(
			key=lambda x: x.get("created_at", ""),
			reverse=True
		)
		
		return jsonify({
			"success": True,
			"data": exceptions
		}), 200
		
	except Exception as e:
		return jsonify({
			"success": False,
			"message": f"Error retrieving exceptions: {str(e)}"
		}), 500


@app.route('/api/exception/<exception_id>', methods=['GET'])
def get_exception_detail(exception_id):
	"""
	Get full details of a specific exception, including similar POCs.
	"""
	try:
		exception_filepath = os.path.join(EXCEPTION_DIR, f"{exception_id}.json")
		
		if not os.path.exists(exception_filepath):
			return jsonify({
				"success": False,
				"message": f"Exception '{exception_id}' not found"
			}), 404
		
		# Read exception data
		with open(exception_filepath, 'r', encoding='utf-8') as f:
			exception_data = json.load(f)
		
		# Ensure title is present in exception data
		if not exception_data.get("title"):
			idea_id = exception_data.get("idea_id", "")
			if idea_id:
				exception_data["title"] = get_idea_title(idea_id)
		
		# Try to find similar POCs
		similar_pocs = []
		if retriever:
			try:
				title = exception_data.get("title", "")
				description = exception_data.get("description", "")
				problem = exception_data.get("problem", "")
				
				# Find similar POCs
				matches = retriever.find_similar_pocs(
					title=title,
					description=description,
					problem=problem,
					score_threshold=0.5,
					top_k=3
				)
				
				# Load full POC details
				for match in matches:
					poc_id = match.get("poc_id")
					poc_data = retriever.get_poc_from_knowledge_base(poc_id)
					if poc_data:
						similar_pocs.append({
							"poc_id": poc_id,
							"similarity_score": match.get("score"),
							"poc_data": poc_data
						})
			except Exception as e:
				print(f"⚠️ Error finding similar POCs: {e}")
		
		return jsonify({
			"success": True,
			"data": {
				"exception": exception_data,
				"similar_pocs": similar_pocs
			}
		}), 200
		
	except Exception as e:
		return jsonify({
			"success": False,
			"message": f"Error retrieving exception details: {str(e)}"
		}), 500


@app.route('/api/exception/<exception_id>', methods=['PUT'])
def update_exception(exception_id):
	"""
	Update exception status and/or comments.
	When status is set to "approved", automatically ingest the POC to Pinecone and save to KB.
	Expects JSON: { "status": "new_status", "notes": "comment text" }
	"""
	try:
		exception_filepath = os.path.join(EXCEPTION_DIR, f"{exception_id}.json")
		
		if not os.path.exists(exception_filepath):
			return jsonify({
				"success": False,
				"message": f"Exception '{exception_id}' not found"
			}), 404
		
		# Read current exception data
		with open(exception_filepath, 'r', encoding='utf-8') as f:
			exception_data = json.load(f)
		
		# Get update data from request
		update_data = request.get_json()
		new_status = update_data.get('status')
		ingestion_status = None
		
		print(f"\n📋 Updating exception {exception_id}")
		print(f"   Received status: {new_status}")
		print(f"   Status type: {type(new_status)}")
		sys.stdout.flush()
		
		# Update status if provided
		if 'status' in update_data:
			exception_data['status'] = update_data['status']
			print(f"   Updated status to: {exception_data['status']}")
		
		# Update notes/comments if provided
		if 'notes' in update_data:
			exception_data['notes'] = update_data['notes']
		
		# Add/update modified timestamp
		exception_data['updated_at'] = datetime.utcnow().isoformat()
		
		# If status is approved, ingest the POC to Pinecone and save to KB
		print(f"   Checking if new_status is 'approved': {new_status and new_status.lower() == 'approved'}")
		if new_status and new_status.lower() == "approved":
			print(f"\n⚙️ Exception approved - Ingesting POC to Pinecone and knowledge base...")
			
			idea_id = exception_data.get("idea_id")
			if idea_id:
				# Load the POC from uploads folder
				poc_file = os.path.join(UPLOADS_DIR, f"{idea_id}.json")
				print(f"   idea_id: {idea_id}")
				print(f"   Looking for POC at: {poc_file}")
				print(f"   File exists: {os.path.exists(poc_file)}")
				
				if os.path.exists(poc_file):
					try:
						with open(poc_file, "r", encoding="utf-8") as f:
							poc_record = json.load(f)
						
						print(f"✅ Loaded POC from uploads: {poc_file}")
						
						# Save to knowledge base
						kb_success, kb_path, kb_msg = save_poc_to_knowledge_base(poc_record)
						print(kb_msg)
						
						# Ingest to Pinecone if embeddings available
						if embeddings_model:
							try:
								print(f"🔄 Ingesting approved POC to Pinecone...")
								
								# Chunk and ingest
								chunks = chunk_documents([poc_record], chunk_size=500, chunk_overlap=20)
								if chunks:
									docsearch = ingest_documents(chunks, index_name="innoscan", embeddings=embeddings_model)
									ingestion_status = f"✅ Approved POC ingested successfully to Pinecone (ID: {idea_id}, {len(chunks)} chunks)"
									print(ingestion_status)
								else:
									ingestion_status = "⚠️ Failed to create chunks for ingestion"
									print(ingestion_status)
							except Exception as e:
								ingestion_status = f"⚠️ Pinecone ingestion failed: {str(e)}"
								print(ingestion_status)
								import traceback
								traceback.print_exc()
						else:
							ingestion_status = "⚠️ Embeddings model not available for Pinecone ingestion"
							print(ingestion_status)
					
					except Exception as e:
						print(f"❌ Error processing approved POC: {str(e)}")
						ingestion_status = f"❌ Error processing approved POC: {str(e)}"
				else:
					print(f"⚠️ POC file not found in uploads: {poc_file}")
					ingestion_status = f"⚠️ POC file not found in uploads"
			else:
				print(f"⚠️ No idea_id found in exception data")
				ingestion_status = "⚠️ No idea_id found in exception data"
		
		# Write updated data back to file
		with open(exception_filepath, 'w', encoding='utf-8') as f:
			json.dump(exception_data, f, indent=2, ensure_ascii=False)
		
		# Build response message
		response_message = "Exception updated successfully"
		if ingestion_status:
			response_message += f" | {ingestion_status}"
		
		return jsonify({
			"success": True,
			"message": response_message,
			"data": exception_data,
			"ingestion_status": ingestion_status
		}), 200
		
	except Exception as e:
		return jsonify({
			"success": False,
			"message": f"Error updating exception: {str(e)}"
		}), 500


@app.errorhandler(404)
def not_found(error):
	return jsonify({
		"success": False,
		"message": "Resource not found"
	}), 404


if __name__ == '__main__':
	try:
		print("🚀 Starting Flask server on http://localhost:5001")
		print("Press Ctrl+C to stop the server gracefully\n")
		
		app.run(
			debug=False,
			port=5001,
			host='127.0.0.1',
			use_reloader=False,
			threaded=True
		)
		
	except KeyboardInterrupt:
		print("\n\n🛑 Server stopped by user")
		sys.exit(0)
	except Exception as e:
		print(f"\n\n❌ Server error: {e}")
		sys.exit(1)
