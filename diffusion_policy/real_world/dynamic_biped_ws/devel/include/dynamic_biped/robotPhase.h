// Generated by gencpp from file dynamic_biped/robotPhase.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_ROBOTPHASE_H
#define DYNAMIC_BIPED_MESSAGE_ROBOTPHASE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace dynamic_biped
{
template <class ContainerAllocator>
struct robotPhase_
{
  typedef robotPhase_<ContainerAllocator> Type;

  robotPhase_()
    : mainPhase(0)
    , subPhase(0)  {
    }
  robotPhase_(const ContainerAllocator& _alloc)
    : mainPhase(0)
    , subPhase(0)  {
  (void)_alloc;
    }



   typedef uint8_t _mainPhase_type;
  _mainPhase_type mainPhase;

   typedef uint8_t _subPhase_type;
  _subPhase_type subPhase;





  typedef boost::shared_ptr< ::dynamic_biped::robotPhase_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::robotPhase_<ContainerAllocator> const> ConstPtr;

}; // struct robotPhase_

typedef ::dynamic_biped::robotPhase_<std::allocator<void> > robotPhase;

typedef boost::shared_ptr< ::dynamic_biped::robotPhase > robotPhasePtr;
typedef boost::shared_ptr< ::dynamic_biped::robotPhase const> robotPhaseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::robotPhase_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::robotPhase_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::robotPhase_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotPhase_<ContainerAllocator2> & rhs)
{
  return lhs.mainPhase == rhs.mainPhase &&
    lhs.subPhase == rhs.subPhase;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::robotPhase_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotPhase_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotPhase_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotPhase_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotPhase_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotPhase_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotPhase_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotPhase_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::robotPhase_<ContainerAllocator> >
{
  static const char* value()
  {
    return "26cd19545acfae4dd7729b92456b2c32";
  }

  static const char* value(const ::dynamic_biped::robotPhase_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x26cd19545acfae4dULL;
  static const uint64_t static_value2 = 0xd7729b92456b2c32ULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::robotPhase_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/robotPhase";
  }

  static const char* value(const ::dynamic_biped::robotPhase_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::robotPhase_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 mainPhase\n"
"uint8 subPhase\n"
;
  }

  static const char* value(const ::dynamic_biped::robotPhase_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::robotPhase_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.mainPhase);
      stream.next(m.subPhase);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct robotPhase_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::robotPhase_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::robotPhase_<ContainerAllocator>& v)
  {
    s << indent << "mainPhase: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.mainPhase);
    s << indent << "subPhase: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.subPhase);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_ROBOTPHASE_H
