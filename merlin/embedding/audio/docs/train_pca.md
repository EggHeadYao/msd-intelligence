# Audio PCA Training Flow

```mermaid
flowchart TD
    A["Prepared audio features"] --> B["Preprocess audio columns"]
    B --> C["Assemble feature vector"]
    C --> D["Standardize features"]
    D --> E["Train PCA model"]
    E --> F["Select embedding dimension"]
    F --> G["Project songs to PCA space"]
    G --> H["L2-normalize embeddings"]
    H --> I["song_embeddings_audio.parquet"]

    E --> J["pca_model"]
    D --> K["scaler_model"]
    B --> L["audio_encoder_metadata.json"]
    F --> L

    classDef main fill:#eef6ff,stroke:#3778c2,color:#111827;
    classDef output fill:#fff7db,stroke:#c58b00,color:#111827;
    class A,B,C,D,E,F,G,H main;
    class I,J,K,L output;
```

- Read the prepared audio feature table.
- Preprocess raw audio fields into numeric model features.
- Assemble and standardize the feature vector.
- Train PCA and select the final embedding dimension.
- Project each song, normalize the embedding, and write the embedding table.
- Save the scaler, PCA model, and encoder metadata for reproducibility.
