# Dependencies and Component Structure

ONTraC has several core dependencies that are automatically installed during the installation process:

```{mermaid}
flowchart LR
subgraph A["ONTraC Dependencies Structure"]
  subgraph B1["Core Dependencies"]
    B11[PyTorch]
    B12["PyTorch Geometric (PyG)"]
    B13(Pandas)
    B14(pyyaml)
    B15(scipy)
  end

  subgraph B2["Analysis Dependencies (Optional)"]
    B21[matplotlib]
    B22[seaborn]
  end

  C1[ONTraC]
  C2["ONTraC analysis"]

  subgraph D["ONTraC Components"]
    subgraph E["Running Components"]
      E1("ONTraC_NN
      (Niche Network)")
      E2("ONTraC_GNN
      (Graph Neural Network)")
      E3("ONTraC_NT
      (Niche Trajectory)")
    end

    subgraph F["Analysis Components"]
      F1("ONTraC_analysis
      (Visualization)")
    end
  end
end
B11 --> C1
B12 --> C1
B13 --> C1
B14 --> C1
B15 --> C1
B12 --> C1
B21 --> C2
B22 --> C2
C1 --> E1
C1 --> E2
C1 --> E3
C1 --> F1
C2 --> F1
```
