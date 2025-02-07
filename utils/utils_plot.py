import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class PlotUtils():

    @staticmethod
    def plot_eval(pred, gt, serials, metrics, csvcol, dstpath):
        x = range(len(gt))
        plt.clf()
        # plt.scatter(x, gt, s=10, c='g', label='Ground Truth')
        # plt.scatter(x, pred, s=10, c='r', label='Predictions')

        # plt.scatter(x, gt,   c='g', label='Ground Truth')
        # plt.scatter(x, pred, c='r', label='Predictions')

        plt.plot(x, gt, c='g', label='Ground Truth', marker='o', linestyle='-')
        plt.plot(x, pred, c='r', label='Predictions', marker='o', linestyle='-')

        n = len(serials)
        for i in range(n):
            plt.annotate(serials[i], (x[i], pred[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.legend()
        plt.grid()

        plt.xlabel('Indices')
        plt.ylabel(csvcol.title())
        #plt.title('Predictions vs Ground Truth\nMAE: {}'.format(round(test_maeloss, 4)))
        ttl = 'Predictions vs Ground Truth\n {}'.format(metrics)
        plt.title(ttl)

        # plt.show()
        plt.savefig(dstpath)

    @staticmethod
    def visualize_embedding(dstpath, embedding,  labels=None , n_components=2, perplexity=30, random_state=42):
        """
        Visualize a PyTorch embedding using t-SNE.

        Parameters:
            embedding (torch.Tensor): The embedding tensor (shape: [n_samples, n_features]).
            labels (list or np.ndarray, optional): Labels for each sample for coloring the visualization.
            n_components (int): Number of components for t-SNE (2 or 3).
            perplexity (float): t-SNE perplexity parameter.
            random_state (int): Random seed for reproducibility.

        Returns:
            None. Displays the plot.
        """
        # Convert PyTorch tensor to NumPy array
        #embedding_np = embedding#.detach().cpu().numpy()

        embedding_np = embedding#.detach().cpu().numpy()
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_embedding = tsne.fit_transform(embedding_np)

        # Visualization
        if n_components == 2:
            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1],
                                  c=labels, cmap='viridis', s=50, alpha=0.7)
            if labels is not None:
                plt.colorbar(scatter, label="Labels")
            plt.title("2D t-SNE Visualization of Embeddings")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.grid(True)
            #plt.show()
            plt.savefig(dstpath)

        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], reduced_embedding[:, 2],
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
            if labels is not None:
                fig.colorbar(scatter, label="Labels")
            ax.set_title("3D t-SNE Visualization of Embeddings")
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_zlabel("t-SNE Component 3")
            plt.show()
            plt.savefig(dstpath)
        else:
            raise ValueError("n_components must be 2 or 3.")

    @staticmethod
    def visualize_embedding_with_attributes(embedding: torch.Tensor,
                                            age, gender, score,
                                            n_components=2, perplexity=30, random_state=42):
        """
        Visualize a PyTorch embedding with additional attributes.

        Parameters:
            embedding (torch.Tensor): The embedding tensor (shape: [n_samples, n_features]).
            age (list or np.ndarray): Ages of subjects.
            gender (list or np.ndarray): Genders of subjects (e.g., 0 for male, 1 for female).
            score (list or np.ndarray): Scores of subjects.
            n_components (int): Number of components for t-SNE (2 or 3).
            perplexity (float): t-SNE perplexity parameter.
            random_state (int): Random seed for reproducibility.

        Returns:
            None. Displays the plot.
        """
        # Convert PyTorch tensor to NumPy array
        embedding_np = embedding.detach().cpu().numpy()

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_embedding = tsne.fit_transform(embedding_np)

        # Normalize age for marker size scaling
        # age_normalized = (age - np.min(age)) / (np.max(age) - np.min(age)) * 50 + 10  # Scale to range [10, 60]
        age_normalized = (age - np.min(age)) / (np.max(age) - np.min(age)) * 50 + 10  # Scale to range [10, 60]

        # Visualization
        if n_components == 2:
            #plt.figure(figsize=(12, 8), dpi=300)
            plt.figure(figsize=(12, 8), dpi=300)
            for g, marker in zip(np.unique(gender), ['o', '^']):  # Assign markers to genders
            # for g, marker in zip(np.unique(gender), ['o', 'v']):  # Assign markers to genders
                idx = np.where(gender == g)

                glabel = "Male"
                if(g == 0):
                    glabel = "Female"

                scatter = plt.scatter(reduced_embedding[idx, 0], reduced_embedding[idx, 1],
                                      c=score[idx], cmap='coolwarm',
                                      # s=age_normalized[idx], alpha=0.7,
                                      s=age_normalized[idx]*15, alpha=0.7,
                                      label=f"{glabel}", marker=marker)

            plt.colorbar(scatter, label="Scores")
            plt.title("2D t-SNE Visualization with Attributes")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.legend(title="")
            plt.grid(True)
            plt.show()

        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            for g, marker in zip(np.unique(gender), ['o', '^']):  # Assign markers to genders
                idx = np.where(gender == g)
                scatter = ax.scatter(reduced_embedding[idx, 0], reduced_embedding[idx, 1], reduced_embedding[idx, 2],
                                     c=score[idx], cmap='coolwarm',
                                     s=age_normalized[idx], alpha=0.7,
                                     label=f"Gender {g}", marker=marker)
            fig.colorbar(scatter, label="Scores")
            ax.set_title("3D t-SNE Visualization with Attributes")
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_zlabel("t-SNE Component 3")
            ax.legend(title="Gender")
            plt.show()
        else:
            raise ValueError("n_components must be 2 or 3.")
