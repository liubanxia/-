# BroadcastModels

This directory is intentionally tiny in source control. CI stages only the hot ReplayKit detector lanes here for the full LiteView package.

Runtime rule: multiple compiled models may exist on disk, but only one custom Core ML detector may be resident in the Broadcast Upload Extension at a time. Heavy Phoenix capability-bank models remain in the main app and are never loaded by ReplayKit.
