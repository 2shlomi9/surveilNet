-- Helpful indexes for query performance

-- Lookup by person
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes WHERE name = 'IX_FaceEmbeddings_PersonID' AND object_id = OBJECT_ID('dbo.FaceEmbeddings')
)
BEGIN
    CREATE INDEX IX_FaceEmbeddings_PersonID ON dbo.FaceEmbeddings(PersonID);
END;
GO

-- Retrieve frames per video quickly
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes WHERE name = 'IX_FaceEmbeddings_VideoFrame' AND object_id = OBJECT_ID('dbo.FaceEmbeddings')
)
BEGIN
    CREATE INDEX IX_FaceEmbeddings_VideoFrame ON dbo.FaceEmbeddings(VideoName, FrameNumber);
END;
GO

-- Time-based
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes WHERE name = 'IX_FaceEmbeddings_CreatedAt' AND object_id = OBJECT_ID('dbo.FaceEmbeddings')
)
BEGIN
    CREATE INDEX IX_FaceEmbeddings_CreatedAt ON dbo.FaceEmbeddings(CreatedAt);
END;
GO
