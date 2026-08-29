#import <Foundation/Foundation.h>

/// Radar/PiP autoload is intentionally disabled in the clean-overlay candidate.
/// The source is kept in the repository for historical comparison, but no launcher card,
/// AVPictureInPictureController, or radar render timer is started at runtime.
@interface LiteViewRadarAutoLoader : NSObject
@end

@implementation LiteViewRadarAutoLoader

+ (void)load {
    // Intentionally empty.
}

@end
