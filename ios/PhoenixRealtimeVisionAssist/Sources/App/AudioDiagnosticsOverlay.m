#import <UIKit/UIKit.h>
#import <notify.h>

static NSString * const LiteViewAudioDiagnosticsNotificationName = @"com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1";
static NSInteger const LiteViewAudioDiagnosticsTag = 84025;
static int LiteViewAudioDiagnosticsToken = -1;

@interface LiteViewAudioDiagnosticsOverlay : NSObject
@end

@implementation LiteViewAudioDiagnosticsOverlay

+ (void)load {
    dispatch_async(dispatch_get_main_queue(), ^{
        notify_register_check(LiteViewAudioDiagnosticsNotificationName.UTF8String,
                              &LiteViewAudioDiagnosticsToken);
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(refreshNow)
                                                     name:UIApplicationDidBecomeActiveNotification
                                                   object:nil];
        [NSTimer scheduledTimerWithTimeInterval:0.5
                                         target:self
                                       selector:@selector(refreshTimer:)
                                       userInfo:nil
                                        repeats:YES];
        [self refreshNow];
    });
}

+ (void)refreshTimer:(NSTimer *)timer {
    if (UIApplication.sharedApplication.applicationState == UIApplicationStateActive) {
        [self refreshNow];
    }
}

+ (UIWindow *)activeWindow {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) { continue; }
        UIWindowScene *windowScene = (UIWindowScene *)scene;
        if (windowScene.activationState != UISceneActivationStateForegroundActive &&
            windowScene.activationState != UISceneActivationStateForegroundInactive) {
            continue;
        }
        for (UIWindow *window in windowScene.windows) {
            if (window.isKeyWindow) { return window; }
        }
        if (windowScene.windows.firstObject) { return windowScene.windows.firstObject; }
    }
    return nil;
}

+ (void)refreshNow {
    UIWindow *window = [self activeWindow];
    if (!window) { return; }

    UILabel *label = (UILabel *)[window viewWithTag:LiteViewAudioDiagnosticsTag];
    if (!label) {
        label = [[UILabel alloc] initWithFrame:CGRectZero];
        label.tag = LiteViewAudioDiagnosticsTag;
        label.translatesAutoresizingMaskIntoConstraints = NO;
        label.numberOfLines = 2;
        label.font = [UIFont monospacedSystemFontOfSize:10.5 weight:UIFontWeightSemibold];
        label.textColor = UIColor.whiteColor;
        label.backgroundColor = [UIColor colorWithWhite:0.03 alpha:0.82];
        label.layer.cornerRadius = 10.0;
        label.layer.masksToBounds = YES;
        label.textAlignment = NSTextAlignmentCenter;
        label.userInteractionEnabled = NO;
        [window addSubview:label];

        UILayoutGuide *safe = window.safeAreaLayoutGuide;
        [NSLayoutConstraint activateConstraints:@[
            [label.topAnchor constraintEqualToAnchor:safe.topAnchor constant:8.0],
            [label.centerXAnchor constraintEqualToAnchor:safe.centerXAnchor],
            [label.widthAnchor constraintLessThanOrEqualToAnchor:safe.widthAnchor constant:-24.0],
            [label.widthAnchor constraintGreaterThanOrEqualToConstant:280.0],
            [label.heightAnchor constraintEqualToConstant:44.0]
        ]];
    }

    label.text = [self diagnosticsText];
    [window bringSubviewToFront:label];
}

+ (NSString *)diagnosticsText {
    uint64_t state = 0;
    BOOL ready = LiteViewAudioDiagnosticsToken >= 0 &&
        notify_get_state(LiteViewAudioDiagnosticsToken, &state) == NOTIFY_STATUS_OK &&
        (state & (UINT64_C(1) << 63)) != 0;

    if (!ready) {
        return @"App 音频诊断 · 等待 ReplayKit audioApp\n不读取麦克风 · 不保存音频";
    }

    uint64_t count = state & UINT64_C(0x0FFF);
    double left = ((state >> 12) & UINT64_C(0x00FF)) / 255.0;
    double right = ((state >> 20) & UINT64_C(0x00FF)) / 255.0;
    double peak = ((state >> 28) & UINT64_C(0x00FF)) / 255.0;
    uint64_t band = (state >> 36) & UINT64_C(0x0007);
    BOOL transient = ((state >> 39) & UINT64_C(1)) != 0;
    uint64_t sampleRateKHz = (state >> 40) & UINT64_C(0x00FF);
    uint64_t channels = (state >> 48) & UINT64_C(0x00FF);
    uint64_t sampleUptimeSlot = (state >> 56) & UINT64_C(0x003F);
    BOOL active = ((state >> 62) & UINT64_C(1)) != 0;
    uint64_t currentUptimeSlot = ((uint64_t)NSProcessInfo.processInfo.systemUptime) & UINT64_C(0x003F);
    uint64_t ageSeconds = (currentUptimeSlot - sampleUptimeSlot) & UINT64_C(0x003F);

    NSString *streamStatus = nil;
    if (active && ageSeconds <= 4) {
        streamStatus = @"持续投递";
    } else if (active) {
        streamStatus = [NSString stringWithFormat:@"已接通 · %llus 未更新", ageSeconds];
    } else {
        streamStatus = @"广播已停止 · 末次汇总";
    }

    return [NSString stringWithFormat:
            @"audioApp %@ · #%llu · %llukHz/%lluch · 频段 %@%@\n"
             "L %.0f%% · R %.0f%% · Peak %.0f%% · 原始音频不落盘",
            streamStatus,
            count,
            sampleRateKHz,
            channels,
            [self bandName:band],
            transient ? @" · 瞬态" : @"",
            left * 100.0,
            right * 100.0,
            peak * 100.0];
}

+ (NSString *)bandName:(uint64_t)band {
    switch (band) {
        case 1: return @"160Hz";
        case 2: return @"400Hz";
        case 3: return @"900Hz";
        case 4: return @"1.8kHz";
        case 5: return @"3.5kHz";
        case 6: return @"7kHz";
        default: return @"未定";
    }
}

@end
