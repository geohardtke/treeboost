//! TRB (TreeBoost) incremental model format
//!
//! A journaled container format supporting incremental training with:
//! - Zero-copy reading via rkyv
//! - O(1) appending (no file rewrite)
//! - Crash recovery (incomplete writes detected)
//! - CRC32 integrity checks per segment
//!
//! # File Layout
//!
//! ```text
//! [MAGIC: "TRB1"]           4 bytes
//! [HEADER_SIZE]             8 bytes, u64 LE
//! [HEADER_JSON]             N bytes
//! [BASE_MODEL_BLOB]         M bytes, rkyv (8-byte aligned)
//! [BASE_CRC32]              4 bytes, u32 LE
//! --- UPDATE SEGMENTS ---
//! [UPDATE_TOTAL_SIZE]       8 bytes, u64 LE
//! [UPDATE_HEADER_SIZE]      8 bytes, u64 LE
//! [UPDATE_HEADER_JSON]      K bytes
//! [UPDATE_BLOB]             L bytes, rkyv (8-byte aligned)
//! [UPDATE_CRC32]            4 bytes, u32 LE
//! ```

use crate::{Result, TreeBoostError};
use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic bytes identifying a TRB file
pub const TRB_MAGIC: &[u8; 4] = b"TRB1";

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Required alignment for rkyv blobs
const RKYV_ALIGNMENT: usize = 8;

/// Header metadata stored as JSON at the start of the file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrbHeader {
    /// Format version for compatibility checking
    pub format_version: u32,
    /// Type of model stored ("universal" or "gbdt")
    pub model_type: String,
    /// Unix timestamp when the file was created
    pub created_at: u64,
    /// Boosting mode (e.g., "PureTree", "LinearThenTree", "RandomForest")
    pub boosting_mode: String,
    /// Number of features in the model
    pub num_features: usize,
    /// Size of the base model blob in bytes
    pub base_blob_size: u64,
    /// User-provided description
    #[serde(default)]
    pub metadata: String,
}

/// Update types that can be appended to a TRB file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateType {
    /// Linear model weights update
    Linear,
    /// Additional trees
    Trees,
    /// Preprocessor state update
    Preprocessor,
    /// Full model snapshot (replaces previous state)
    Snapshot,
}

/// Header for an update segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrbUpdateHeader {
    /// Type of update
    pub update_type: UpdateType,
    /// Unix timestamp when the update was created
    pub created_at: u64,
    /// Number of rows used to train this update
    pub rows_trained: usize,
    /// User-provided description
    #[serde(default)]
    pub description: String,
}

/// A parsed segment from a TRB file
#[derive(Debug)]
pub enum TrbSegment {
    /// The base model segment
    Base {
        header: TrbHeader,
        blob: Vec<u8>,
    },
    /// An update segment
    Update {
        header: TrbUpdateHeader,
        blob: Vec<u8>,
    },
}

/// Writer for creating and appending to TRB files
pub struct TrbWriter {
    file: File,
    header: TrbHeader,
}

impl TrbWriter {
    /// Create a new TRB file with a base model
    ///
    /// # Arguments
    /// * `path` - Path to create the file at
    /// * `header` - Metadata about the base model
    /// * `base_blob` - The serialized base model (rkyv bytes)
    pub fn new(
        path: impl AsRef<Path>,
        mut header: TrbHeader,
        base_blob: &[u8],
    ) -> Result<Self> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;

        // Acquire exclusive lock
        file.try_lock_exclusive().map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to acquire file lock: {}", e))
        })?;

        // Update header with actual blob size
        header.base_blob_size = base_blob.len() as u64;

        // Write magic
        file.write_all(TRB_MAGIC)?;

        // Serialize header to JSON
        let header_json = serde_json::to_vec(&header)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize header: {}", e)))?;

        // Write header size
        file.write_all(&(header_json.len() as u64).to_le_bytes())?;

        // Write header JSON
        file.write_all(&header_json)?;

        // Calculate padding for 8-byte alignment
        let current_pos = 4 + 8 + header_json.len();
        let padding = alignment_padding(current_pos);
        if padding > 0 {
            file.write_all(&vec![0u8; padding])?;
        }

        // Write base blob
        file.write_all(base_blob)?;

        // Write CRC32
        let crc = crc32fast::hash(base_blob);
        file.write_all(&crc.to_le_bytes())?;

        file.flush()?;

        Ok(Self { file, header })
    }

    /// Append an update segment to the TRB file
    ///
    /// # Arguments
    /// * `update_header` - Metadata about the update
    /// * `update_blob` - The serialized update data (rkyv bytes)
    pub fn append_update(
        &mut self,
        update_header: &TrbUpdateHeader,
        update_blob: &[u8],
    ) -> Result<()> {
        // Seek to end
        self.file.seek(SeekFrom::End(0))?;

        // Serialize update header to JSON
        let header_json = serde_json::to_vec(update_header)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize update header: {}", e)))?;

        // Calculate padding for blob alignment
        // Position after: total_size(8) + header_size(8) + header_json
        let header_section_size = 8 + 8 + header_json.len();
        let padding = alignment_padding(header_section_size);
        let padded_header_json_len = header_json.len() + padding;

        // Total size = header_size(8) + padded_header_json + blob + crc(4)
        let total_size = 8 + padded_header_json_len + update_blob.len() + 4;

        // Write total size
        self.file.write_all(&(total_size as u64).to_le_bytes())?;

        // Write header size (includes padding)
        self.file.write_all(&(padded_header_json_len as u64).to_le_bytes())?;

        // Write header JSON + padding
        self.file.write_all(&header_json)?;
        if padding > 0 {
            self.file.write_all(&vec![0u8; padding])?;
        }

        // Write blob
        self.file.write_all(update_blob)?;

        // Write CRC32
        let crc = crc32fast::hash(update_blob);
        self.file.write_all(&crc.to_le_bytes())?;

        self.file.flush()?;

        Ok(())
    }

    /// Get the header of this TRB file
    pub fn header(&self) -> &TrbHeader {
        &self.header
    }
}

impl Drop for TrbWriter {
    fn drop(&mut self) {
        // Release lock on drop
        let _ = self.file.unlock();
    }
}

/// Reader for TRB files
pub struct TrbReader {
    file: File,
    header: TrbHeader,
    base_blob_offset: u64,
}

impl TrbReader {
    /// Open a TRB file for reading
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path.as_ref())?;

        // Acquire shared lock
        file.try_lock_shared().map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to acquire file lock: {}", e))
        })?;

        // Read and validate magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != TRB_MAGIC {
            return Err(TreeBoostError::Serialization(format!(
                "Invalid TRB magic: expected {:?}, got {:?}",
                TRB_MAGIC, magic
            )));
        }

        // Read header size
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        // Read header JSON
        let mut header_json = vec![0u8; header_size];
        file.read_exact(&mut header_json)?;

        let header: TrbHeader = serde_json::from_slice(&header_json)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to parse header: {}", e)))?;

        // Calculate base blob offset (after magic + header_size + header + padding)
        let current_pos = 4 + 8 + header_size;
        let padding = alignment_padding(current_pos);
        let base_blob_offset = (current_pos + padding) as u64;

        Ok(Self {
            file,
            header,
            base_blob_offset,
        })
    }

    /// Get the header of this TRB file
    pub fn header(&self) -> &TrbHeader {
        &self.header
    }

    /// Read the base model blob, validating CRC
    pub fn read_base_blob(&mut self) -> Result<Vec<u8>> {
        let blob_size = self.header.base_blob_size as usize;

        // Seek to base blob
        self.file.seek(SeekFrom::Start(self.base_blob_offset))?;

        // Read blob
        let mut blob = vec![0u8; blob_size];
        self.file.read_exact(&mut blob)?;

        // Read and validate CRC
        let mut crc_bytes = [0u8; 4];
        self.file.read_exact(&mut crc_bytes)?;
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = crc32fast::hash(&blob);

        if stored_crc != computed_crc {
            return Err(TreeBoostError::Serialization(format!(
                "Base blob CRC mismatch: stored={:#x}, computed={:#x}",
                stored_crc, computed_crc
            )));
        }

        Ok(blob)
    }

    /// Iterate over update segments
    ///
    /// Returns an iterator yielding `(TrbUpdateHeader, Vec<u8>)` for each valid update.
    /// Invalid or incomplete updates are skipped with a warning.
    pub fn iter_updates(&mut self) -> Result<Vec<(TrbUpdateHeader, Vec<u8>)>> {
        let mut updates = Vec::new();

        // Position after base blob + CRC
        let mut pos = self.base_blob_offset + self.header.base_blob_size + 4;
        self.file.seek(SeekFrom::Start(pos))?;

        let file_size = self.file.seek(SeekFrom::End(0))?;
        self.file.seek(SeekFrom::Start(pos))?;

        let mut segment_index = 0;
        while pos < file_size {
            // Check if we have enough bytes for total_size
            if pos + 8 > file_size {
                eprintln!(
                    "Warning: Incomplete update segment {} at offset {} (truncated total_size)",
                    segment_index, pos
                );
                break;
            }

            // Read total size
            let mut total_size_bytes = [0u8; 8];
            if self.file.read_exact(&mut total_size_bytes).is_err() {
                eprintln!(
                    "Warning: Failed to read update segment {} at offset {}",
                    segment_index, pos
                );
                break;
            }
            let total_size = u64::from_le_bytes(total_size_bytes) as usize;

            // Check if we have the full segment
            if pos + 8 + total_size as u64 > file_size {
                eprintln!(
                    "Warning: Incomplete update segment {} at offset {} (expected {} bytes, have {})",
                    segment_index, pos, total_size, file_size - pos - 8
                );
                break;
            }

            // Read header size
            let mut header_size_bytes = [0u8; 8];
            self.file.read_exact(&mut header_size_bytes)?;
            let header_size = u64::from_le_bytes(header_size_bytes) as usize;

            // Read header JSON (may include padding)
            let mut header_json = vec![0u8; header_size];
            self.file.read_exact(&mut header_json)?;

            // Trim trailing zeros (padding)
            let json_end = header_json.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
            let header_json_trimmed = &header_json[..json_end];

            let update_header: TrbUpdateHeader = match serde_json::from_slice(header_json_trimmed) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse update header at segment {}: {}",
                        segment_index, e
                    );
                    // Skip to next segment
                    pos += 8 + total_size as u64;
                    self.file.seek(SeekFrom::Start(pos))?;
                    segment_index += 1;
                    continue;
                }
            };

            // Calculate blob size: total_size - header_size(8) - header_json - crc(4)
            let blob_size = total_size - 8 - header_size - 4;

            // Read blob
            let mut blob = vec![0u8; blob_size];
            self.file.read_exact(&mut blob)?;

            // Read and validate CRC
            let mut crc_bytes = [0u8; 4];
            self.file.read_exact(&mut crc_bytes)?;
            let stored_crc = u32::from_le_bytes(crc_bytes);
            let computed_crc = crc32fast::hash(&blob);

            if stored_crc != computed_crc {
                eprintln!(
                    "Warning: Update segment {} CRC mismatch (stored={:#x}, computed={:#x})",
                    segment_index, stored_crc, computed_crc
                );
                // Stop processing - CRC failure means chain is broken
                break;
            }

            updates.push((update_header, blob));
            pos += 8 + total_size as u64;
            segment_index += 1;
        }

        Ok(updates)
    }

    /// Load all segments (base + updates) as a vector
    pub fn load_all_segments(&mut self) -> Result<Vec<TrbSegment>> {
        let mut segments = Vec::new();

        // Load base
        let base_blob = self.read_base_blob()?;
        segments.push(TrbSegment::Base {
            header: self.header.clone(),
            blob: base_blob,
        });

        // Load updates
        for (update_header, blob) in self.iter_updates()? {
            segments.push(TrbSegment::Update {
                header: update_header,
                blob,
            });
        }

        Ok(segments)
    }
}

impl Drop for TrbReader {
    fn drop(&mut self) {
        // Release lock on drop
        let _ = self.file.unlock();
    }
}

/// Open an existing TRB file for appending updates
pub fn open_for_append(path: impl AsRef<Path>) -> Result<TrbWriter> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path.as_ref())?;

    // Acquire exclusive lock
    file.try_lock_exclusive().map_err(|e| {
        TreeBoostError::Serialization(format!("Failed to acquire file lock: {}", e))
    })?;

    // Read and validate magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != TRB_MAGIC {
        return Err(TreeBoostError::Serialization(format!(
            "Invalid TRB magic: expected {:?}, got {:?}",
            TRB_MAGIC, magic
        )));
    }

    // Read header size
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    // Read header JSON
    let mut header_json = vec![0u8; header_size];
    file.read_exact(&mut header_json)?;

    let header: TrbHeader = serde_json::from_slice(&header_json)
        .map_err(|e| TreeBoostError::Serialization(format!("Failed to parse header: {}", e)))?;

    Ok(TrbWriter { file, header })
}

/// Calculate padding needed to reach alignment
fn alignment_padding(current_pos: usize) -> usize {
    let remainder = current_pos % RKYV_ALIGNMENT;
    if remainder == 0 {
        0
    } else {
        RKYV_ALIGNMENT - remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn create_test_header() -> TrbHeader {
        TrbHeader {
            format_version: FORMAT_VERSION,
            model_type: "universal".to_string(),
            created_at: current_timestamp(),
            boosting_mode: "PureTree".to_string(),
            num_features: 10,
            base_blob_size: 0, // Will be updated by TrbWriter
            metadata: "Test model".to_string(),
        }
    }

    #[test]
    fn test_trb_write_and_read_base() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create test blob (simulating rkyv serialized data)
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let header = create_test_header();

        // Write
        let writer = TrbWriter::new(&path, header.clone(), &base_blob).unwrap();
        drop(writer); // Release lock

        // Read
        let mut reader = TrbReader::open(&path).unwrap();

        // Verify magic (file exists and is valid)
        assert_eq!(reader.header().format_version, FORMAT_VERSION);
        assert_eq!(reader.header().model_type, "universal");
        assert_eq!(reader.header().num_features, 10);
        assert_eq!(reader.header().base_blob_size, base_blob.len() as u64);

        // Verify blob
        let loaded_blob = reader.read_base_blob().unwrap();
        assert_eq!(loaded_blob, base_blob);
    }

    #[test]
    fn test_trb_append_update() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = create_test_header();
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        // Get initial size
        let initial_size = std::fs::metadata(&path).unwrap().len();

        // Append update
        let mut writer = open_for_append(&path).unwrap();
        let update_header = TrbUpdateHeader {
            update_type: UpdateType::Trees,
            created_at: current_timestamp(),
            rows_trained: 500,
            description: "Update 1".to_string(),
        };
        let update_blob = vec![10u8, 20, 30, 40];
        writer.append_update(&update_header, &update_blob).unwrap();
        drop(writer);

        // Verify file size increased
        let new_size = std::fs::metadata(&path).unwrap().len();
        assert!(new_size > initial_size);

        // Read and verify
        let mut reader = TrbReader::open(&path).unwrap();
        let segments = reader.load_all_segments().unwrap();

        assert_eq!(segments.len(), 2); // Base + 1 update

        // Verify base
        assert!(
            matches!(&segments[0], TrbSegment::Base { .. }),
            "Expected base segment at index 0"
        );
        if let TrbSegment::Base { blob, .. } = &segments[0] {
            assert_eq!(blob, &base_blob);
        }

        // Verify update
        assert!(
            matches!(&segments[1], TrbSegment::Update { .. }),
            "Expected update segment at index 1"
        );
        if let TrbSegment::Update { header, blob } = &segments[1] {
            assert_eq!(header.update_type, UpdateType::Trees);
            assert_eq!(header.rows_trained, 500);
            assert_eq!(blob, &update_blob);
        }
    }

    #[test]
    fn test_trb_corrupt_recovery() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base + update
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = create_test_header();
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        let mut writer = open_for_append(&path).unwrap();
        let update_header = TrbUpdateHeader {
            update_type: UpdateType::Trees,
            created_at: current_timestamp(),
            rows_trained: 500,
            description: "Update 1".to_string(),
        };
        let update_blob = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        writer.append_update(&update_header, &update_blob).unwrap();
        drop(writer);

        // Truncate file to simulate crash (remove last 10 bytes)
        let file_size = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(file_size - 10).unwrap();
        drop(file);

        // Read - should recover base, warn about incomplete update
        let mut reader = TrbReader::open(&path).unwrap();

        // Base should load fine
        let base = reader.read_base_blob().unwrap();
        assert_eq!(base, base_blob);

        // Updates should be empty (truncated update ignored)
        let updates = reader.iter_updates().unwrap();
        assert!(updates.is_empty(), "Truncated update should be ignored");
    }

    #[test]
    fn test_trb_crc_detects_corruption() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = create_test_header();
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        // Corrupt a byte in the base blob (not truncate, just flip a bit)
        let mut file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        // Calculate where the blob starts
        file.seek(SeekFrom::Start(4)).unwrap(); // Skip magic
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes).unwrap();
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;
        let current_pos = 4 + 8 + header_size;
        let padding = alignment_padding(current_pos);
        let blob_offset = current_pos + padding;

        // Flip a bit in the blob
        file.seek(SeekFrom::Start(blob_offset as u64)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF; // Flip all bits
        file.seek(SeekFrom::Start(blob_offset as u64)).unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        // Try to read - should detect CRC mismatch
        let mut reader = TrbReader::open(&path).unwrap();
        let result = reader.read_base_blob();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("CRC mismatch"));
    }

    #[test]
    fn test_trb_multiple_updates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = create_test_header();
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        // Append 5 updates
        for i in 0..5 {
            let mut writer = open_for_append(&path).unwrap();
            let update_header = TrbUpdateHeader {
                update_type: UpdateType::Trees,
                created_at: current_timestamp(),
                rows_trained: (i + 1) * 100,
                description: format!("Update {}", i + 1),
            };
            let update_blob = vec![(i + 10) as u8; 8];
            writer.append_update(&update_header, &update_blob).unwrap();
            drop(writer);
        }

        // Read and verify
        let mut reader = TrbReader::open(&path).unwrap();
        let segments = reader.load_all_segments().unwrap();

        assert_eq!(segments.len(), 6); // Base + 5 updates

        // Verify each update
        for (i, segment) in segments.iter().enumerate().skip(1) {
            assert!(
                matches!(segment, TrbSegment::Update { .. }),
                "Expected update segment at index {}",
                i
            );
            if let TrbSegment::Update { header, blob } = segment {
                assert_eq!(header.rows_trained, i * 100);
                assert_eq!(blob, &vec![(i + 9) as u8; 8]);
            }
        }
    }

    #[test]
    fn test_trb_update_crc_validation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base + 2 updates with large blobs so we can reliably corrupt the blob area
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = create_test_header();
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        for i in 0..2 {
            let mut writer = open_for_append(&path).unwrap();
            let update_header = TrbUpdateHeader {
                update_type: UpdateType::Trees,
                created_at: current_timestamp(),
                rows_trained: (i + 1) * 100,
                description: format!("U{}", i + 1), // Short description
            };
            // Large blob (128 bytes) to ensure we have room to corrupt
            let update_blob = vec![(i + 10) as u8; 128];
            writer.append_update(&update_header, &update_blob).unwrap();
            drop(writer);
        }

        // Read to get the exact structure
        let reader = TrbReader::open(&path).unwrap();
        let base_end = reader.base_blob_offset + reader.header.base_blob_size + 4;
        drop(reader);

        // Calculate where update 1's blob starts
        // Layout: total_size(8) + header_size(8) + header_json(padded) + blob + crc(4)
        // Header JSON for "U1" is roughly 80 bytes, padded to 8-byte alignment
        // We'll corrupt a byte well into the blob area (offset 100 into the update segment)
        let mut file = OpenOptions::new().read(true).write(true).open(&path).unwrap();

        // Read the header size to know where blob starts
        file.seek(SeekFrom::Start(base_end)).unwrap();
        let mut total_size_bytes = [0u8; 8];
        file.read_exact(&mut total_size_bytes).unwrap();
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes).unwrap();
        let header_size = u64::from_le_bytes(header_size_bytes) as u64;

        // Blob starts at: base_end + 8 (total_size) + 8 (header_size) + header_size
        let blob_start = base_end + 8 + 8 + header_size;
        // Corrupt a byte in the middle of the blob
        let corrupt_offset = blob_start + 64;

        file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF;
        file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        // Read - base should be fine, update 1 should fail CRC, update 2 should NOT be loaded
        let mut reader = TrbReader::open(&path).unwrap();
        let base = reader.read_base_blob().unwrap();
        assert_eq!(base, base_blob);

        let updates = reader.iter_updates().unwrap();
        // Update 1 CRC fails, chain breaks, update 2 not loaded
        assert!(updates.is_empty(), "Corrupted update should break the chain");
    }

    #[test]
    fn test_trb_unknown_json_fields_ignored() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create base with extra field in header
        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];

        // Manually write with extra field
        let mut file = File::create(&path).unwrap();
        file.write_all(TRB_MAGIC).unwrap();

        let header_json = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "model_type": "universal",
            "created_at": current_timestamp(),
            "boosting_mode": "PureTree",
            "num_features": 10,
            "base_blob_size": base_blob.len(),
            "metadata": "Test",
            "future_field": "some_value", // Unknown field
            "another_future": 42
        });
        let header_bytes = serde_json::to_vec(&header_json).unwrap();

        file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header_bytes).unwrap();

        // Padding
        let current_pos = 4 + 8 + header_bytes.len();
        let padding = alignment_padding(current_pos);
        if padding > 0 {
            file.write_all(&vec![0u8; padding]).unwrap();
        }

        // Blob + CRC
        file.write_all(&base_blob).unwrap();
        let crc = crc32fast::hash(&base_blob);
        file.write_all(&crc.to_le_bytes()).unwrap();
        drop(file);

        // Read - should not error on unknown fields
        let mut reader = TrbReader::open(&path).unwrap();
        assert_eq!(reader.header().num_features, 10);
        let blob = reader.read_base_blob().unwrap();
        assert_eq!(blob, base_blob);
    }

    #[test]
    fn test_trb_rkyv_alignment() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.trb");

        // Create header that results in non-aligned position
        let header = TrbHeader {
            format_version: FORMAT_VERSION,
            model_type: "u".to_string(), // Short to test padding
            created_at: 12345,
            boosting_mode: "P".to_string(),
            num_features: 1,
            base_blob_size: 0,
            metadata: "".to_string(),
        };

        let base_blob = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let writer = TrbWriter::new(&path, header, &base_blob).unwrap();
        drop(writer);

        // Verify blob starts at 8-byte aligned offset
        let mut reader = TrbReader::open(&path).unwrap();
        assert_eq!(reader.base_blob_offset % 8, 0, "Base blob should be 8-byte aligned");

        let blob = reader.read_base_blob().unwrap();
        assert_eq!(blob, base_blob);
    }
}
