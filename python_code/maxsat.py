import os
import subprocess
import tempfile
def MaxSAT(hard_clauses, soft_clauses):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.wcnf') as file:
        wcnf_file = file.name
        for clause in hard_clauses:
            file.write(f"h {' '.join(map(str, clause))} 0\n")
        
        for weight, clause in soft_clauses:
            file.write(f"{weight} {' '.join(map(str, clause))} 0\n")
        file.flush()
        
    try:
        print(f"Running tt-open-wbo-inc on {wcnf_file}...")
        result = subprocess.run(
            ["./tt-open-wbo-inc-Glucose4_1_static", wcnf_file], 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout
        print(f"Solver output preview: {output[:200]}...")
        if "OPTIMUM FOUND" in output:
            print("Optimal solution found!")
        else:
            print("No optimal solution found.")
            print(f"Solver output: {output}")

        os.unlink(wcnf_file)
        return output
    
    except Exception as e:
        print(f"Error running Max-SAT solver: {e}")
        if os.path.exists(wcnf_file):
            os.unlink(wcnf_file)
        return None

def main():
    MaxSAT([[1,2,3,4]],[(1,[2,3])])
    
if __name__ == "__main__":
    main()
