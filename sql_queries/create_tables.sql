-- Create core tables for Face Recognition pipeline (SQL Server)

IF OBJECT_ID('dbo.FaceGallery', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.FaceGallery (
        PersonID UNIQUEIDENTIFIER NOT NULL PRIMARY KEY,
        AverageEmbedding VARBINARY(2048) NOT NULL,        -- 512 * 4 bytes float32
        CreatedAt DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
        UpdatedAt DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
    );
END;
GO

IF OBJECT_ID('dbo.FaceEmbeddings', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.FaceEmbeddings (
        Id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        VideoName NVARCHAR(255) NOT NULL,
        FrameNumber INT NOT NULL,
        PersonID UNIQUEIDENTIFIER NOT NULL,
        Embedding VARBINARY(2048) NOT NULL,               -- 512 * 4 bytes float32
        BBox_X1 INT NOT NULL,
        BBox_Y1 INT NOT NULL,
        BBox_X2 INT NOT NULL,
        BBox_Y2 INT NOT NULL,
        CreatedAt DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_FaceEmbeddings_FaceGallery
            FOREIGN KEY (PersonID) REFERENCES dbo.FaceGallery(PersonID)
            ON UPDATE NO ACTION ON DELETE NO ACTION
    );
END;
GO
