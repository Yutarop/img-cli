package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const pythonScript = `# hybrid_image_clustering.py - Embedded version
import os
import subprocess
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import time
import sys
import argparse

# Try to import TensorFlow/Keras with fallback
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

class HybridImageClustering:
    def __init__(self, input_folder, output_folder, n_clusters=5, max_workers=8, 
                 use_pca=True, variance_ratio=0.95, visualize=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_clusters = n_clusters
        self.max_workers = max_workers
        self.use_pca = use_pca
        self.variance_ratio = variance_ratio
        self.visualize = visualize
        self.model = None
        self.features = None
        self.image_paths = []
        self.cluster_labels = None
        
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # Go binaries
        self.go_loader_path = "image_loader"
        self.go_mover_path = "image_mover"
        
        if os.name == 'nt':  # Windows
            self.go_loader_path += ".exe"
            self.go_mover_path += ".exe"

    def check_dependencies(self):
        """Check if required dependencies are available"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
        
        try:
            import sklearn
            import PIL
            import numpy
            import matplotlib
        except ImportError as e:
            raise ImportError(f"Required dependency missing: {e}")
        
        print("All required dependencies are available")

    def compile_go_programs(self):
        """Compile Go programs if source files are available"""
        print("Checking for Go programs...")
        
        # Check if Go is available
        try:
            result = subprocess.run(["go", "version"], capture_output=True, text=True, check=True)
            print(f"Go version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Go not available, will use Python-only mode")
            return False

        # Check for Go source files
        if not os.path.exists("image_loader.go") or not os.path.exists("image_mover.go"):
            print("Go source files not found, using Python-only mode")
            return False

        # Initialize go.mod if it doesn't exist
        if not os.path.exists("go.mod"):
            print("Creating go.mod...")
            try:
                subprocess.run(["go", "mod", "init", "image-clustering"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to create go.mod: {e}")
                return False

        # Install dependencies
        try:
            subprocess.run(["go", "get", "github.com/nfnt/resize"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to install Go dependencies")

        # Compile image_loader.go
        try:
            result = subprocess.run(
                ["go", "build", "-o", self.go_loader_path, "image_loader.go"], 
                capture_output=True, text=True, check=True
            )
            print("image_loader.go compiled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile image_loader.go: {e.stderr}")
            return False

        # Compile image_mover.go
        try:
            result = subprocess.run(
                ["go", "build", "-o", self.go_mover_path, "image_mover.go"], 
                capture_output=True, text=True, check=True
            )
            print("image_mover.go compiled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile image_mover.go: {e.stderr}")
            return False

        return True

    def load_model(self):
        """Load ResNet50 model"""
        print("Loading ResNet50 model...")
        try:
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print("Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load ResNet50 model: {e}")

    def extract_features_with_go(self):
        """Extract features using Go for parallel image loading"""
        print("Starting hybrid feature extraction (Go + Python)...")
        start_time = time.time()

        temp_dir = os.path.join(self.output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Execute Go program for parallel image loading
            cmd = [
                self.go_loader_path,
                "-input", self.input_folder,
                "-output", temp_dir,
                "-workers", str(self.max_workers)
            ]

            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                raise Exception(f"Go program failed with code {result.returncode}: {result.stderr}")

            if not result.stdout.strip():
                raise Exception("No output from Go program")

            # Parse Go program output
            try:
                output_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"Raw Go output: {result.stdout}")
                raise Exception(f"Failed to parse Go output as JSON: {e}")

            if output_data.get("successful_images", 0) == 0:
                errors = output_data.get("errors", [])
                print("Go program processing errors:")
                for error in errors[:5]:
                    print(f"  {error}")
                raise Exception("Go program failed to process any images")

            self.image_paths = output_data["image_paths"]
            loaded_images_info = output_data["loaded_images"]

            print(f"Go program processed {len(loaded_images_info)} images successfully")
            print(f"Go processing time: {output_data['processing_time_seconds']:.2f} seconds")

            # Extract features using Python
            print("Extracting features with ResNet50...")
            features_list = []

            for i, img_info in enumerate(loaded_images_info):
                try:
                    processed_path = img_info["processed_path"]

                    if not os.path.exists(processed_path):
                        print(f"Warning: Processed file not found: {processed_path}")
                        continue

                    # Read binary data
                    data = np.fromfile(processed_path, dtype=np.float64)
                    expected_size = 224 * 224 * 3
                    
                    if len(data) != expected_size:
                        print(f"Warning: Invalid data size: {len(data)} (expected: {expected_size})")
                        continue

                    # Reshape to image format
                    img_array = data.reshape(1, 224, 224, 3)
                    img_array = preprocess_input(img_array)

                    # Extract features
                    features = self.model.predict(img_array, verbose=0)
                    features_list.append(features.flatten())

                    if (i + 1) % 20 == 0 or (i + 1) == len(loaded_images_info):
                        print(f"Progress: {i + 1}/{len(loaded_images_info)} features extracted")

                except Exception as e:
                    print(f"Error processing {img_info.get('original_path', 'unknown')}: {e}")
                    continue

            # Clean up temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            if len(features_list) == 0:
                raise Exception("No valid features could be extracted")

            self.features = np.array(features_list)
            # Keep only paths for successfully processed images
            self.image_paths = [info["original_path"] for info in loaded_images_info 
                              if not info.get("error")][:len(features_list)]

            total_time = time.time() - start_time
            print(f"Hybrid feature extraction completed in {total_time:.2f} seconds")
            print(f"Feature matrix shape: {self.features.shape}")

        except Exception as e:
            print(f"Hybrid extraction failed: {e}")
            raise

    def extract_features_python_only(self):
        """Extract features using Python only"""
        print("Extracting features using Python only...")
        start_time = time.time()
        
        # Collect image files
        self.image_paths = []
        for root, dirs, files in os.walk(self.input_folder):
            for filename in files:
                if filename.lower().endswith(self.supported_formats):
                    self.image_paths.append(os.path.join(root, filename))

        if not self.image_paths:
            raise ValueError(f"No supported images found in {self.input_folder}")

        print(f"Found {len(self.image_paths)} images")

        features_list = []
        failed_count = 0

        for i, img_path in enumerate(self.image_paths):
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Extract features
                features = self.model.predict(img_array, verbose=0)
                features_list.append(features.flatten())

                if (i + 1) % 10 == 0 or (i + 1) == len(self.image_paths):
                    print(f"Progress: {i + 1}/{len(self.image_paths)} images processed")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failed_count += 1
                continue

        if len(features_list) == 0:
            raise Exception("No valid features could be extracted")

        # Remove failed image paths
        valid_paths = []
        for i, path in enumerate(self.image_paths):
            if i - failed_count < len(features_list):
                valid_paths.append(path)

        self.features = np.array(features_list)
        self.image_paths = valid_paths

        total_time = time.time() - start_time
        print(f"Python-only feature extraction completed in {total_time:.2f} seconds")
        print(f"Successfully processed {len(features_list)} out of {len(self.image_paths) + failed_count} images")
        print(f"Feature matrix shape: {self.features.shape}")

    def perform_clustering(self):
        """Perform K-means clustering"""
        print("Starting clustering...")
        
        features_for_clustering = self.features

        # Apply PCA if enabled
        if self.use_pca and self.features.shape[1] > 50:
            n_samples, n_features = self.features.shape
            max_components = min(n_samples - 1, n_features - 1)

            if max_components > 50:
                print("Analyzing optimal PCA components...")
                temp_pca = PCA()
                temp_pca.fit(self.features)
                cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_ratio >= self.variance_ratio) + 1
                n_components = min(n_components, max_components)
            else:
                n_components = max_components

            if n_components > 1:
                print(f"PCA dimensionality reduction: {n_features} -> {n_components}")
                pca = PCA(n_components=n_components)
                features_for_clustering = pca.fit_transform(self.features)
                print(f"Cumulative variance ratio retained: {pca.explained_variance_ratio_.sum():.3f}")
            else:
                print("Skipping PCA: insufficient components")
        elif not self.use_pca:
            print("PCA disabled")

        # Adjust cluster count if necessary
        actual_clusters = min(self.n_clusters, len(self.image_paths))
        if actual_clusters != self.n_clusters:
            print(f"Adjusting cluster count from {self.n_clusters} to {actual_clusters}")
            self.n_clusters = actual_clusters

        # Perform K-means clustering
        print(f"Running K-means clustering (k={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(features_for_clustering)

        print("Clustering completed successfully!")

        # Display cluster statistics
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} images")

    def move_images_with_go(self):
        """Move images using Go for parallel processing"""
        print("Starting parallel image moving with Go...")
        start_time = time.time()

        # Create cluster data file for Go program
        cluster_data = {
            "image_paths": self.image_paths,
            "cluster_labels": self.cluster_labels.tolist(),
            "output_folder": self.output_folder,
            "n_clusters": self.n_clusters
        }

        cluster_file = os.path.join(self.output_folder, "cluster_data.json")
        with open(cluster_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)

        try:
            # Execute Go mover program
            cmd = [
                self.go_mover_path,
                "-data", cluster_file,
                "-workers", str(self.max_workers)
            ]

            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise Exception(f"Go mover failed with code {result.returncode}: {result.stderr}")

            if not result.stdout.strip():
                raise Exception("No output from Go mover")

            # Parse results
            try:
                output_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"Raw Go output: {result.stdout}")
                raise Exception(f"Failed to parse Go mover output: {e}")

            print(f"Successfully moved {output_data['moved_count']} images")
            print(f"Go moving time: {output_data['processing_time_seconds']:.2f} seconds")

            if output_data.get('errors'):
                print("Some files had errors during moving:")
                for error in output_data['errors'][:5]:
                    print(f"  {error}")

        except Exception as e:
            print(f"Go-based image moving failed: {e}")
            raise

        # Clean up cluster data file
        try:
            os.remove(cluster_file)
        except:
            pass

        total_time = time.time() - start_time
        print(f"Image moving completed in {total_time:.2f} seconds")

    def move_images_python(self):
        """Move images using Python"""
        print("Moving images to cluster folders...")
        import shutil
        
        os.makedirs(self.output_folder, exist_ok=True)
        moved_count = 0
        error_count = 0

        for img_path, cluster_id in zip(self.image_paths, self.cluster_labels):
            try:
                cluster_folder = os.path.join(self.output_folder, f'cluster_{cluster_id}')
                os.makedirs(cluster_folder, exist_ok=True)

                filename = os.path.basename(img_path)
                destination = os.path.join(cluster_folder, filename)

                # Handle duplicate filenames
                counter = 1
                original_destination = destination
                while os.path.exists(destination):
                    name, ext = os.path.splitext(original_destination)
                    destination = f"{name}_{counter}{ext}"
                    counter += 1

                shutil.copy2(img_path, destination)
                moved_count += 1

            except Exception as e:
                print(f"Error moving {img_path}: {e}")
                error_count += 1

        print(f"Image moving completed: {moved_count} images moved, {error_count} errors")

    def visualize_clusters(self, max_images_per_cluster=5):
        """Create cluster visualization"""
        if not self.visualize:
            print("Visualization disabled")
            return
            
        print("Creating cluster visualization...")
        
        try:
            # Calculate figure size
            fig_width = min(15, max_images_per_cluster * 3)
            fig_height = self.n_clusters * 3
            
            fig, axes = plt.subplots(self.n_clusters, max_images_per_cluster,
                                    figsize=(fig_width, fig_height))
            fig.suptitle('Image Clustering Results', fontsize=16)

            # Handle single cluster case
            if self.n_clusters == 1:
                axes = axes.reshape(1, -1)
            elif max_images_per_cluster == 1:
                axes = axes.reshape(-1, 1)

            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
                
                for i in range(max_images_per_cluster):
                    if self.n_clusters == 1 and max_images_per_cluster == 1:
                        ax = axes
                    elif self.n_clusters == 1:
                        ax = axes[0, i]
                    elif max_images_per_cluster == 1:
                        ax = axes[cluster_id, 0]
                    else:
                        ax = axes[cluster_id, i]

                    if i < len(cluster_indices):
                        img_path = self.image_paths[cluster_indices[i]]
                        try:
                            img = Image.open(img_path)
                            ax.imshow(img)
                            if i == 0:  # Only show title on first image
                                ax.set_title(f'Cluster {cluster_id} ({len(cluster_indices)} images)', 
                                           fontsize=10)
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Load Error', ha='center', va='center', 
                                   transform=ax.transAxes, fontsize=8)
                    else:
                        ax.axis('off')

                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(self.output_folder, 'cluster_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved: {viz_path}")

        except Exception as e:
            print(f"Visualization failed: {e}")

    def save_results(self):
        """Save clustering results and features"""
        try:
            # Save features and metadata
            results_file = os.path.join(self.output_folder, 'clustering_results.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'features': self.features,
                    'image_paths': self.image_paths,
                    'cluster_labels': self.cluster_labels,
                    'n_clusters': self.n_clusters,
                    'config': {
                        'use_pca': self.use_pca,
                        'variance_ratio': self.variance_ratio,
                        'input_folder': self.input_folder,
                        'output_folder': self.output_folder
                    }
                }, f)
            
            print(f"Results saved: {results_file}")
            
            # Save summary statistics
            summary_file = os.path.join(self.output_folder, 'summary.json')
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            summary = {
                'total_images': len(self.image_paths),
                'n_clusters': self.n_clusters,
                'feature_dimensions': self.features.shape[1] if self.features is not None else 0,
                'cluster_distribution': {f'cluster_{i}': int(count) for i, count in zip(unique, counts)},
                'config': {
                    'use_pca': self.use_pca,
                    'variance_ratio': self.variance_ratio,
                    'input_folder': self.input_folder,
                    'output_folder': self.output_folder
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Summary saved: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Could not save results: {e}")

    def run_hybrid_pipeline(self):
        """Run the complete hybrid (Go + Python) pipeline"""
        print("Running hybrid clustering pipeline (Go + Python)...")
        print("=" * 60)
        
        try:
            # Check dependencies
            self.check_dependencies()
            
            # Try to compile Go programs
            go_available = self.compile_go_programs()
            
            # Load ML model
            self.load_model()
            
            # Feature extraction
            if go_available and os.path.exists(self.go_loader_path):
                try:
                    self.extract_features_with_go()
                except Exception as e:
                    print(f"Hybrid feature extraction failed: {e}")
                    print("Falling back to Python-only feature extraction...")
                    self.extract_features_python_only()
            else:
                self.extract_features_python_only()
            
            # Clustering
            self.perform_clustering()
            
            # Image organization
            if go_available and os.path.exists(self.go_mover_path):
                try:
                    self.move_images_with_go()
                except Exception as e:
                    print(f"Go-based image moving failed: {e}")
                    print("Falling back to Python-based image moving...")
                    self.move_images_python()
            else:
                self.move_images_python()
            
            # Visualization and results
            self.visualize_clusters()
            self.save_results()
            
            print("=" * 60)
            print("All processing completed successfully!")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)

    def run_python_only_pipeline(self):
        """Run Python-only clustering pipeline"""
        print("Running Python-only clustering pipeline...")
        print("=" * 60)
        
        try:
            self.check_dependencies()
            self.load_model()
            self.extract_features_python_only()
            self.perform_clustering()
            self.move_images_python()
            self.visualize_clusters()
            self.save_results()
            
            print("=" * 60)
            print("All processing completed successfully!")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Hybrid Image Clustering Tool')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for clustered images')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--no-pca', action='store_true', help='Disable PCA dimensionality reduction')
    parser.add_argument('--variance-ratio', type=float, default=0.95, help='PCA variance ratio to retain')
    parser.add_argument('--no-viz', action='store_true', help='Disable cluster visualization')
    parser.add_argument('--python-only', action='store_true', help='Force Python-only mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.clusters < 1 or args.clusters > 100:
        print("Error: Cluster count must be between 1 and 100")
        sys.exit(1)
    
    if args.variance_ratio < 0.1 or args.variance_ratio > 1.0:
        print("Error: Variance ratio must be between 0.1 and 1.0")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize clustering system
    clustering = HybridImageClustering(
        input_folder=args.input,
        output_folder=args.output,
        n_clusters=args.clusters,
        max_workers=args.workers,
        use_pca=not args.no_pca,
        variance_ratio=args.variance_ratio,
        visualize=not args.no_viz
    )
    
    # Run appropriate pipeline
    if args.python_only:
        clustering.run_python_only_pipeline()
    else:
        clustering.run_hybrid_pipeline()

if __name__ == "__main__":
    main()
`

func checkPythonEnvironment() error {
	// Check if Python is available
	cmd := exec.Command(pythonPath, "--version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("python not found at '%s': %w", pythonPath, err)
	}

	if verbose {
		fmt.Printf("Python version: %s", string(output))
	}

	// Check required Python packages
	requiredPackages := []string{
		"tensorflow", "sklearn", "PIL", "numpy", "matplotlib"
	}

	for _, pkg := range requiredPackages {
		cmd := exec.Command(pythonPath, "-c", fmt.Sprintf("import %s", pkg))
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("required Python package '%s' is not installed. Please run: pip install %s", pkg, pkg)
		}
	}

	if verbose {
		fmt.Println("All required Python packages are available")
	}

	return nil
}

func runPythonScript() error {
	// Create temporary directory for the Python script
	tempDir, err := os.MkdirTemp("", "imgcli_*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tempDir)

	// Write Python script to temporary file
	scriptPath := filepath.Join(tempDir, "clustering.py")
	if err := os.WriteFile(scriptPath, []byte(pythonScript), 0644); err != nil {
		return fmt.Errorf("failed to write Python script: %w", err)
	}

	// Copy Go source files to temp directory if they exist
	goFiles := []string{"image_loader.go", "image_mover.go"}
	for _, goFile := range goFiles {
		if _, err := os.Stat(goFile); err == nil {
			srcContent, err := os.ReadFile(goFile)
			if err != nil {
				if verbose {
					fmt.Printf("Warning: Could not read %s: %v\n", goFile, err)
				}
				continue
			}
			
			dstPath := filepath.Join(tempDir, goFile)
			if err := os.WriteFile(dstPath, srcContent, 0644); err != nil {
				if verbose {
					fmt.Printf("Warning: Could not write %s to temp dir: %v\n", goFile, err)
				}
			}
		}
	}

	// Build Python command arguments
	args := []string{scriptPath}
	args = append(args, "--input", inputDir)
	args = append(args, "--output", outputDir)
	args = append(args, "--clusters", fmt.Sprintf("%d", clusters))
	args = append(args, "--workers", fmt.Sprintf("%d", workers))

	if noPCA {
		args = append(args, "--no-pca")
	}
	if varRatio != 0.95 {
		args = append(args, "--variance-ratio", fmt.Sprintf("%.3f", varRatio))
	}
	if noVisualize {
		args = append(args, "--no-viz")
	}

	// Set working directory for Go compilation
	originalDir, _ := os.Getwd()
	if err := os.Chdir(tempDir); err != nil {
		return fmt.Errorf("failed to change to temp directory: %w", err)
	}
	defer os.Chdir(originalDir)

	// Execute Python script
	if verbose {
		fmt.Printf("Executing: %s %s\n", pythonPath, strings.Join(args, " "))
	}
	
	cmd := exec.Command(pythonPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	
	// Set environment variables
	cmd.Env = os.Environ()
	
	// Add timeout for long-running operations
	done := make(chan error, 1)
	go func() {
		done <- cmd.Run()
	}()

	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("Python script execution failed: %w", err)
		}
	case <-time.After(30 * time.Minute): // 30 minute timeout
		if err := cmd.Process.Kill(); err != nil {
			fmt.Printf("Warning: failed to kill process: %v\n", err)
		}
		return fmt.Errorf("Python script execution timed out after 30 minutes")
	}

	return nil
}

func runPythonClustering() error {
	fmt.Println("Starting image clustering pipeline...")
	
	start := time.Now()
	if err := runPythonScript(); err != nil {
		return fmt.Errorf("clustering pipeline failed: %w", err)
	}
	
	elapsed := time.Since(start)
	fmt.Printf("Pipeline completed in %s\n", elapsed.Round(time.Second))
	
	return nil
}package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const pythonScript = `# hybrid_image_clustering.py - Embedded version
import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import time
import sys
import argparse

class HybridImageClustering:
    def __init__(self, input_folder, output_folder, n_clusters=5, max_workers=8, use_pca=True, variance_ratio=0.95, visualize=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_clusters = n_clusters
        self.max_workers = max_workers
        self.use_pca = use_pca
        self.variance_ratio = variance_ratio
        self.visualize = visualize
        self.model = None
        self.features = None
        self.image_paths = []
        self.cluster_labels = None
        
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # Go binaries
        self.go_loader_path = "image_loader"
        self.go_mover_path = "image_mover"
        
        if os.name == 'nt':  # Windows
            self.go_loader_path += ".exe"
            self.go_mover_path += ".exe"

    def compile_go_programs(self):
        print("Compiling Go programs...")
        try:
            go_version = subprocess.run(["go", "version"], capture_output=True, text=True, check=True)
            print(f"Go version: {go_version.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("Go is not installed or not in PATH")

        if not os.path.exists("go.mod"):
            print("Creating go.mod...")
            subprocess.run(["go", "mod", "init", "image-clustering"], check=True)

        try:
            subprocess.run(["go", "get", "github.com/nfnt/resize"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to install dependencies, continuing...")

        # Compile image_loader.go
        try:
            subprocess.run(["go", "build", "-o", self.go_loader_path, "image_loader.go"], 
                         capture_output=True, text=True, check=True)
            print("✓ image_loader.go compiled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile image_loader.go: {e.stderr}")
            raise Exception("Failed to compile Go programs")

        # Compile image_mover.go
        try:
            subprocess.run(["go", "build", "-o", self.go_mover_path, "image_mover.go"], 
                         capture_output=True, text=True, check=True)
            print("✓ image_mover.go compiled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile image_mover.go: {e.stderr}")
            raise Exception("Failed to compile Go programs")

    def load_model(self):
        print("Loading ResNet50 model...")
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("Model loaded successfully")

    def extract_features_python_only(self):
        print("Extracting features using Python only...")
        self.image_paths = []
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(self.supported_formats):
                self.image_paths.append(os.path.join(self.input_folder, filename))

        if not self.image_paths:
            raise ValueError(f"No supported images found in {self.input_folder}")

        print(f"Found {len(self.image_paths)} images")

        features_list = []
        for i, img_path in enumerate(self.image_paths):
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                features = self.model.predict(img_array, verbose=0)
                features_list.append(features.flatten())

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(self.image_paths)} features extracted")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if len(features_list) == 0:
            raise Exception("No valid features could be extracted")

        self.features = np.array(features_list)
        self.image_paths = self.image_paths[:len(features_list)]  # Match valid features
        print(f"Feature extraction complete. Shape: {self.features.shape}")

    def perform_clustering(self):
        print("Starting clustering...")
        features_for_clustering = self.features

        if self.use_pca:
            n_samples, n_features = self.features.shape
            max_components = min(n_samples, n_features) - 1

            if max_components > 50:
                temp_pca = PCA()
                temp_pca.fit(self.features)
                cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_ratio >= self.variance_ratio) + 1
                n_components = min(n_components, max_components)
            else:
                n_components = max_components

            if n_components > 1:
                print(f"PCA dimensionality reduction: {n_features} -> {n_components}")
                pca = PCA(n_components=n_components)
                features_for_clustering = pca.fit_transform(self.features)
                print(f"Cumulative variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

        actual_clusters = min(self.n_clusters, len(self.image_paths))
        if actual_clusters != self.n_clusters:
            print(f"Adjusting cluster count from {self.n_clusters} to {actual_clusters}")
            self.n_clusters = actual_clusters

        print(f"Running K-means clustering (k={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(features_for_clustering)

        print("✓ Clustering complete!")

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nImages per cluster:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} images")

    def move_images_python(self):
        print("Moving images to cluster folders...")
        import shutil
        
        os.makedirs(self.output_folder, exist_ok=True)
        moved_count = 0

        for img_path, cluster_id in zip(self.image_paths, self.cluster_labels):
            try:
                cluster_folder = os.path.join(self.output_folder, f'cluster_{cluster_id}')
                os.makedirs(cluster_folder, exist_ok=True)

                filename = os.path.basename(img_path)
                destination = os.path.join(cluster_folder, filename)

                shutil.copy2(img_path, destination)
                moved_count += 1

            except Exception as e:
                print(f"Error moving {img_path}: {e}")

        print(f"Image moving complete: {moved_count} images moved")

    def visualize_clusters(self, max_images_per_cluster=5):
        if not self.visualize:
            return
            
        print("Creating cluster visualization...")
        
        fig, axes = plt.subplots(self.n_clusters, max_images_per_cluster,
                                figsize=(15, 3 * self.n_clusters))
        fig.suptitle('Image Clustering Results', fontsize=16)

        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_images = min(len(cluster_indices), max_images_per_cluster)

            for i in range(max_images_per_cluster):
                if self.n_clusters == 1:
                    ax = axes[i] if max_images_per_cluster > 1 else axes
                else:
                    ax = axes[cluster_id, i] if max_images_per_cluster > 1 else axes[cluster_id]

                if i < cluster_images:
                    img_path = self.image_paths[cluster_indices[i]]
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(f'Cluster {cluster_id}')
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...',
                                ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        viz_path = os.path.join(self.output_folder, 'cluster_visualization.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {viz_path}")

    def run_python_only_pipeline(self):
        print("Running Python-only clustering pipeline...")
        print("=" * 60)
        
        self.load_model()
        self.extract_features_python_only()
        self.perform_clustering()
        self.move_images_python()
        
        if self.visualize:
            try:
                self.visualize_clusters()
            except Exception as e:
                print(f"Visualization failed but processing is complete: {e}")

        print("=" * 60)
        print("All processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Hybrid Image Clustering')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--no-pca', action='store_true', help='Disable PCA')
    parser.add_argument('--variance-ratio', type=float, default=0.95, help='PCA variance ratio')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    clustering = HybridImageClustering(
        args.input, 
        args.output, 
        args.clusters, 
        args.workers,
        use_pca=not args.no_pca,
        variance_ratio=args.variance_ratio,
        visualize=not args.no_viz
    )
    
    # Try hybrid approach first, fallback to Python-only
    try:
        clustering.compile_go_programs()
        # If Go compilation succeeds, we could run hybrid, but for simplicity run Python-only
        clustering.run_python_only_pipeline()
    except Exception as e:
        print(f"Go compilation failed: {e}")
        print("Running Python-only mode...")
        clustering.run_python_only_pipeline()

if __name__ == "__main__":
    main()
`

func runPythonScript() error {
	// Create temporary Python script file
	tempDir, err := os.MkdirTemp("", "imgcli_*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tempDir)

	scriptPath := filepath.Join(tempDir, "clustering.py")
	if err := os.WriteFile(scriptPath, []byte(pythonScript), 0644); err != nil {
		return fmt.Errorf("failed to write Python script: %w", err)
	}

	// Build Python command arguments
	args := []string{scriptPath}
	args = append(args, "--input", inputDir)
	args = append(args, "--output", outputDir)
	args = append(args, "--clusters", fmt.Sprintf("%d", clusters))
	args = append(args, "--workers", fmt.Sprintf("%d", workers))

	if noPCA {
		args = append(args, "--no-pca")
	}
	if varRatio != 0.95 {
		args = append(args, "--variance-ratio", fmt.Sprintf("%.2f", varRatio))
	}
	if noVisualize {
		args = append(args, "--no-viz")
	}

	// Execute Python script
	fmt.Printf("Executing: %s %s\n", pythonPath, strings.Join(args, " "))
	
	cmd := exec.Command(pythonPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir, _ = os.Getwd() // Set working directory

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("Python script execution failed: %w", err)
	}

	fmt.Printf("\n✅ Clustering complete! Results saved to: %s\n", outputDir)
	return nil
}