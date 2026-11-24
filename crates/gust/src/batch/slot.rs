//! Slots: a stage that can be replaced between frames (slice s2d). Frames in flight share the batch's buffers (D258); a
//! replaced stage's buffers live until the frames begun before the replacement have completed.
