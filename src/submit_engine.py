#!/usr/bin/env python3
"""
WRF Model Submission Engine for HPC using Slurm
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import f90nml
import math


class SubmitEngine:
    """Engine for managing WRF model submissions via Slurm"""
    
    def __init__(self, work_dir: str = "."):
        """
        Initialize the submission engine
        
        Args:
            work_dir: Working directory containing WRF files
        """
        self.work_dir = Path(work_dir).resolve()
        self.meta_file = self.work_dir / ".meta"
        self.lock_file = self.work_dir / ".lock"
        self.record_file = self.work_dir / "submit_record.txt"
        self.namelist_original = self.work_dir / "namelist.input.original"
        self.namelist_input = self.work_dir / "namelist.input"
        
    def parse_namelist(self) -> Dict[str, Any]:
        """Parse the original namelist file"""
        if not self.namelist_original.exists():
            raise FileNotFoundError(f"Namelist file not found: {self.namelist_original}")
        
        nml = f90nml.read(self.namelist_original)
        return nml
    
    def calculate_total_runs(self, nml: Dict[str, Any]) -> tuple:
        """
        Calculate total number of runs needed
        
        Returns:
            tuple: (num_runs, start_time, end_time, run_length_timedelta)
        """
        time_control = nml['time_control']
        
        # Extract start time
        start_time = datetime(
            year=time_control['start_year'][0],
            month=time_control['start_month'][0],
            day=time_control['start_day'][0],
            hour=time_control['start_hour'][0],
            minute=time_control['start_minute'][0],
            second=time_control['start_second'][0]
        )
        
        # Extract end time
        end_time = datetime(
            year=time_control['end_year'][0],
            month=time_control['end_month'][0],
            day=time_control['end_day'][0],
            hour=time_control['end_hour'][0],
            minute=time_control['end_minute'][0],
            second=time_control['end_second'][0]
        )
        
        # Extract run length
        run_length = timedelta(
            days=time_control.get('run_days', 0),
            hours=time_control.get('run_hours', 0),
            minutes=time_control.get('run_minutes', 0),
            seconds=time_control.get('run_seconds', 0)
        )
        
        # Calculate total simulation time
        total_time = end_time - start_time
       
        # Calculate number of runs
        num_runs = math.ceil(total_time.total_seconds() / run_length.total_seconds())
       
        if num_runs <= 0:
            raise Exception("Computed number of runs is less than or equal to zero. Please check your simulation time.")
 
        return num_runs, start_time, end_time, run_length
    
    def generate_expected_files(self, num_runs: int, start_time: datetime, 
                               run_length: timedelta) -> List[List[str]]:
        """
        Generate list of expected output files for each run
        
        Args:
            num_runs: Total number of runs
            start_time: Simulation start time
            run_length: Length of each run
            
        Returns:
            List of lists, where each inner list contains expected files for that run
        """
        expected_files = []
        
        for run_idx in range(num_runs):
            run_end_time = start_time + run_length * (run_idx + 1)
            
            # Generate expected WRF output filenames
            # Format: wrfout_d01_YYYY-MM-DD_HH:MM:SS
            files_for_run = []
            
            # Typically WRF outputs one file per domain
            # Adjust domain count as needed (here assuming domain 01)
            domain_str = "d01"
            time_str = run_end_time.strftime("%Y-%m-%d_%H:%M:%S")
            output_file = f"wrfout_{domain_str}_{time_str}"
            files_for_run.append(output_file)
            
            # Can add additional expected files like restart files
            restart_file = f"wrfrst_{domain_str}_{time_str}"
            files_for_run.append(restart_file)
            
            expected_files.append(files_for_run)
        
        return expected_files
    
    def generate_meta(self, force: bool = False) -> None:
        """
        Generate metadata file for job submissions
        
        Args:
            force: If True, overwrite existing meta file without prompt
        """
        if self.meta_file.exists() and not force:
            response = input(f"Meta file {self.meta_file} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Meta generation cancelled.")
                return
        
        print("Generating metadata...")
        
        # Parse namelist
        nml = self.parse_namelist()
        
        # Calculate runs
        num_runs, start_time, end_time, run_length = self.calculate_total_runs(nml)
        
        # Generate expected files
        expected_files = self.generate_expected_files(num_runs, start_time, run_length)
        
        # Create metadata
        meta = {
            "num_runs": num_runs,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "run_length_seconds": run_length.total_seconds(),
            "expected_files": expected_files,
            "created_at": datetime.now().isoformat()
        }
        
        # Write metadata
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Meta file generated: {self.meta_file}")
        print(f"Total runs planned: {num_runs}")
        print(f"Simulation period: {start_time} to {end_time}")
    
    def load_meta(self) -> Dict[str, Any]:
        """Load metadata from file"""
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_file}. Run with --gen-meta first.")
        
        with open(self.meta_file, 'r') as f:
            return json.load(f)
    
    def check_progress(self) -> int:
        """
        Check current progress by verifying expected files
        
        Returns:
            Current run index (0-based). If all files for run N exist, returns N+1
        """
        meta = self.load_meta()
        expected_files = meta['expected_files']
        
        for run_idx, files in enumerate(expected_files):
            # Check if all files exist for this run
            all_exist = all((self.work_dir / f).exists() for f in files)
            if not all_exist:
                return run_idx
        
        # All runs completed
        return len(expected_files)
    
    def create_lock(self, job_id: str, run_idx: int) -> None:
        """
        Create lock file with job information
        
        Args:
            job_id: Slurm job ID
            run_idx: Current run index
        """
        lock_data = {
            "job_id": job_id,
            "run_idx": run_idx,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid()
        }
        
        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
    
    def remove_lock(self, force: bool = False) -> None:
        """
        Remove lock file
        
        Args:
            force: If True, remove without prompt
        """
        if not self.lock_file.exists():
            print("No lock file found.")
            return
        
        if not force:
            response = input(f"Remove lock file {self.lock_file}? (y/n): ")
            if response.lower() != 'y':
                print("Lock removal cancelled.")
                return
        
        self.lock_file.unlink()
        print(f"Lock file removed: {self.lock_file}")
    
    def load_lock(self) -> Optional[Dict[str, Any]]:
        """Load lock file if it exists"""
        if not self.lock_file.exists():
            return None
        
        with open(self.lock_file, 'r') as f:
            return json.load(f)
    
    def is_job_running(self, job_id: str) -> bool:
        """
        Check if a Slurm job is still running
        
        Args:
            job_id: Slurm job ID
            
        Returns:
            True if job is running or pending, False otherwise
        """
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return len(result.stdout.strip()) > 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def check_status(self) -> None:
        """Check and display current status"""
        print("=== Job Status ===")
        
        # Check if meta exists
        if not self.meta_file.exists():
            print("No meta file found. Run with --gen-meta first.")
            return
        
        meta = self.load_meta()
        current_run = self.check_progress()
        total_runs = meta['num_runs']
        
        print(f"Progress: {current_run}/{total_runs} runs completed")
        
        # Check lock
        lock = self.load_lock()
        if lock:
            job_id = lock['job_id']
            run_idx = lock['run_idx']
            timestamp = lock['timestamp']
            
            if self.is_job_running(job_id):
                print(f"Status: RUNNING")
                print(f"Job ID: {job_id}")
                print(f"Current run: {run_idx + 1}/{total_runs}")
                print(f"Started: {timestamp}")
            else:
                print(f"Status: NOT RUNNING (but lock exists)")
                print(f"Last job ID: {job_id}")
                print(f"Last run: {run_idx + 1}/{total_runs}")
                print(f"Timestamp: {timestamp}")
                response = input("Remove lock file? (y/n): ")
                if response.lower() == 'y':
                    self.remove_lock(force=True)
        else:
            print(f"Status: NOT RUNNING")
            if current_run < total_runs:
                print(f"Ready to submit run {current_run + 1}/{total_runs}")
            else:
                print("All runs completed!")
    
    def cancel_job(self) -> None:
        """Cancel the current running job"""
        lock = self.load_lock()
        if not lock:
            print("No lock file found. No job to cancel.")
            return
        
        job_id = lock['job_id']
        
        if not self.is_job_running(job_id):
            print(f"Job {job_id} is not running.")
            return
        
        try:
            subprocess.run(['scancel', job_id], check=True)
            print(f"Job {job_id} cancelled.")
            self.remove_lock(force=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to cancel job {job_id}: {e}")
    
    def reset(self) -> None:
        """Reset by cancelling job and removing meta"""
        print("Resetting submission engine...")
        
        # Cancel job if running
        lock = self.load_lock()
        if lock:
            self.cancel_job()
        
        # Remove meta file
        if self.meta_file.exists():
            self.meta_file.unlink()
            print(f"Meta file removed: {self.meta_file}")
        
        print("Reset complete.")
    
    def clear_record(self) -> None:
        """Remove submission record file"""
        if self.record_file.exists():
            self.record_file.unlink()
            print(f"Record file removed: {self.record_file}")
        else:
            print("No record file found.")
    
    def update_namelist(self, run_idx: int) -> None:
        """
        Update namelist.input for the current run
        
        Args:
            run_idx: Current run index (0-based)
        """
        meta = self.load_meta()
        nml = self.parse_namelist()
        
        # Calculate start and end times for this run
        base_start = datetime.fromisoformat(meta['start_time'])
        run_length = timedelta(seconds=meta['run_length_seconds'])
        
        run_start = base_start + run_length * run_idx
        run_end = run_start + run_length
        
        # Update time_control section
        time_control = nml['time_control']
        
        # Update start time
        time_control['start_year'][0] = run_start.year
        time_control['start_month'][0] = run_start.month
        time_control['start_day'][0] = run_start.day
        time_control['start_hour'][0] = run_start.hour
        time_control['start_minute'][0] = run_start.minute
        time_control['start_second'][0] = run_start.second
        
        # Update end time
        time_control['end_year'][0] = run_end.year
        time_control['end_month'][0] = run_end.month
        time_control['end_day'][0] = run_end.day
        time_control['end_hour'][0] = run_end.hour
        time_control['end_minute'][0] = run_end.minute
        time_control['end_second'][0] = run_end.second
        
        # Set restart flag (restart for runs after the first)
        time_control['restart'] = run_idx > 0
        
        # Write updated namelist
        nml.write(self.namelist_input, force=True)
        print(f"Updated namelist for run {run_idx + 1}: {run_start} to {run_end}")
    
    def append_record(self, job_id: str, run_idx: int) -> None:
        """
        Append submission information to record file
        
        Args:
            job_id: Slurm job ID
            run_idx: Current run index
        """
        timestamp = datetime.now().isoformat()
        record_line = f"{timestamp} | Run {run_idx + 1} | Job ID: {job_id}\n"
        
        with open(self.record_file, 'a') as f:
            f.write(record_line)
    
    def submit(self, sbatch_script: str = "submit.sh") -> None:
        """
        Submit the next job in the sequence
        
        Args:
            sbatch_script: Path to the sbatch submission script
        """
        # Check for lock
        if self.lock_file.exists():
            lock = self.load_lock()
            if self.is_job_running(lock['job_id']):
                print(f"Job {lock['job_id']} is already running.")
                return
            else:
                print("Lock file exists but job is not running. Removing lock.")
                self.remove_lock(force=True)
        
        # Check meta
        if not self.meta_file.exists():
            print("No meta file found. Run with --gen-meta first.")
            return
        
        # Check progress
        meta = self.load_meta()
        current_run = self.check_progress()
        total_runs = meta['num_runs']
        
        if current_run >= total_runs:
            print("All runs completed!")
            return
        
        print(f"Submitting run {current_run + 1}/{total_runs}")
        
        # Update namelist
        self.update_namelist(current_run)
        
        # Submit job
        sbatch_path = self.work_dir / sbatch_script
        if not sbatch_path.exists():
            print(f"Sbatch script not found: {sbatch_path}")
            return
        
        try:
            result = subprocess.run(
                ['sbatch', str(sbatch_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.work_dir
            )
            
            # Extract job ID from output (format: "Submitted batch job 12345")
            output = result.stdout.strip()
            job_id = output.split()[-1]
            
            print(f"Job submitted: {job_id}")
            
            # Create lock
            self.create_lock(job_id, current_run)
            
            # Append to record
            self.append_record(job_id, current_run)
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job: {e}")
            print(f"Error output: {e.stderr}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="WRF Model Submission Engine for HPC using Slurm"
    )
    
    parser.add_argument(
        '--gen-meta',
        action='store_true',
        help="Generate metadata file from namelist"
    )
    
    parser.add_argument(
        '--submit',
        action='store_true',
        help="Submit the next job in the sequence"
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help="Check current status"
    )
    
    parser.add_argument(
        '--unlock',
        action='store_true',
        help="Remove lock file (with prompt)"
    )
    
    parser.add_argument(
        '--force-unlock',
        action='store_true',
        help="Remove lock file without prompt"
    )
    
    parser.add_argument(
        '--cancel-job',
        action='store_true',
        help="Cancel current running job"
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help="Cancel job and remove metadata"
    )
    
    parser.add_argument(
        '--clear-record',
        action='store_true',
        help="Remove submission record file"
    )
    
    parser.add_argument(
        '--work-dir',
        type=str,
        default=".",
        help="Working directory (default: current directory)"
    )
    
    parser.add_argument(
        '--sbatch-script',
        type=str,
        default="submit.sh",
        help="Sbatch script filename (default: submit.sh)"
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = SubmitEngine(work_dir=args.work_dir)
    
    # Execute commands
    try:
        if args.gen_meta:
            engine.generate_meta()
        elif args.submit:
            engine.submit(sbatch_script=args.sbatch_script)
        elif args.check:
            engine.check_status()
        elif args.unlock:
            engine.remove_lock(force=False)
        elif args.force_unlock:
            engine.remove_lock(force=True)
        elif args.cancel_job:
            engine.cancel_job()
        elif args.reset:
            engine.reset()
        elif args.clear_record:
            engine.clear_record()
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
